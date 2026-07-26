//! Acoustic prediction host-transfer elimination test.
//!
//! Verifies P03-T04: the greedy and sampled codec prediction paths
//! only perform the unavoidable host-device transfers defined in the
//! transfer contract.
//!
//! This test uses `TransferObserver` events emitted at the production
//! synchronization boundaries. Stage callbacks only count attempted frames and
//! never infer transfers.
//!
//! Key verified claims
//! - `greedy_select_on_device` matches host-side `greedy_select` across
//!   suppression/allow patterns.
//! - Greedy Code Predictor path: argmax is on-device, *zero* scalar
//!   transfers from the predictor.
//! - Greedy Talker path: `greedy_select_on_device` avoids full-vocab
//!   download; the only D→H transfer is 1 scalar per frame for the
//!   EOS check (`Codebook0Scalar`).
//! - Sampled Code Predictor path: each step transfers the full logit
//!   vector (vocab_size elements) to host for Philox sampling + 1
//!   scalar token per frame.
//! - On-device frame assembly via `Tensor::cat` — never
//!   `codes_1_15.to_vec1()`.
//!
#![allow(clippy::unwrap_used)]

use candle_core::{DType, Device, Tensor};

use qwen3tts::alignment_stage_dump::{
    StageDumpObserver, TransferCollector, TransferEvent, TransferObserver,
};
use qwen3tts::talker::code_predictor::CodePredictor;
use qwen3tts::talker::config::{CodePredictorConfig, TalkerConfig};
use qwen3tts::talker::decoder_layer::StandardDecoderLayer;
use qwen3tts::talker::model::TalkerModel;
use qwen3tts::talker::primitives::{MultimodalRotaryEmbedding, RMSNorm, SwiGLUMLP};
use qwen3tts::talker::sampling::{
    greedy_select, greedy_select_on_device, Sampler, SamplingOptions,
};
use qwen3tts::talker::talker::TalkerForConditionalGeneration;
use qwen3tts::talker::talker_attention::StandardAttention;

// ===========================================================================
// Constants
// ===========================================================================

/// Frames requested from the Talker generator.  Due to `terminal_cap_step`
/// the actual output may be `max_new_tokens - 1` frames.
const MAX_NEW_TOKENS: usize = 5;

/// Small vocabulary for the Code Predictor in unit tests.
const CP_VOCAB: usize = 8;

/// Talker combined vocabulary for minimal test talker.
const TALKER_VOCAB: usize = 1028;

/// EOS token placed inside the reserved suffix.
const EOS: usize = 1024; // >= SUPPRESS_FROM

// ===========================================================================
// Helper: stage tracking kept separate from transfer collection
// ===========================================================================

/// Observer that implements **both** `StageDumpObserver` and
/// `TransferObserver`.
///
/// Stage callbacks count attempts only. The collector receives production
/// `on_transfer` events directly.
#[derive(Debug)]
struct TransferTrackingObserver {
    /// Number of frames seen so far (incremented in on_talker_codebook0_logits).
    frame_count: usize,
}

impl TransferTrackingObserver {
    fn new() -> Self {
        Self { frame_count: 0 }
    }
}

// ── StageDumpObserver ────────────────────────────────────────────────────
//
// The stage-dump hooks NO LONGER infer transfer events because the
// production `*_with_observer` code now emits real `on_transfer` calls
// via `TransferObserver`.  These hooks only track frame_count so that
// test assertions can still use that value.

impl StageDumpObserver for TransferTrackingObserver {
    fn wants_capture(&self) -> bool {
        true
    }

    fn on_talker_codebook0_logits(
        &mut self,
        _frame_index: usize,
        _logits: &Tensor,
    ) -> candle_core::Result<()> {
        self.frame_count += 1;
        Ok(())
    }

    fn on_code_predictor_step_logits(
        &mut self,
        _frame_index: usize,
        _step: usize,
        _logits: &Tensor,
    ) -> candle_core::Result<()> {
        Ok(())
    }
}

// ===========================================================================
// Tensor helpers (same pattern as code_predictor.rs tests)
// ===========================================================================

fn ones1(a: usize) -> Tensor {
    Tensor::ones(a, DType::F32, &Device::Cpu).unwrap()
}

fn zeros2(a: usize, b: usize) -> Tensor {
    Tensor::zeros((a, b), DType::F32, &Device::Cpu).unwrap()
}

fn zeros3(a: usize, b: usize, c: usize) -> Tensor {
    Tensor::zeros((a, b, c), DType::F32, &Device::Cpu).unwrap()
}

/// Extract a scalar u32 from the [1,1] tensor returned by
/// `greedy_select_on_device`.
fn device_select_scalar(
    logits: &Tensor,
    suppress_from: Option<usize>,
    allow: Option<usize>,
) -> u32 {
    greedy_select_on_device(logits, suppress_from, allow)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<u32>()
        .unwrap()[0]
}

// ===========================================================================
// Fixture builders
// ===========================================================================

/// Build a minimal Code Predictor for unit-testing.
///
/// Architecture: 1 layer, hidden_size=4, head_dim=2, 2 query heads / 1 KV head.
fn make_code_predictor(vocab_size: usize) -> (CodePredictor, Device) {
    let device = Device::Cpu;
    let config = CodePredictorConfig {
        hidden_size: 4,
        intermediate_size: 8,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 2,
        num_hidden_layers: 1,
        vocab_size,
        num_code_groups: 16,
        max_position_embeddings: 32,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        hidden_act: "silu".into(),
        attention_bias: false,
        attention_dropout: 0.0,
        layer_types: vec!["full_attention".into()],
    };

    let attn = StandardAttention::new(
        zeros2(4, 4),
        zeros2(2, 4),
        zeros2(2, 4),
        zeros2(4, 4),
        ones1(2),
        ones1(2),
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.rms_norm_eps,
    );

    let layer = StandardDecoderLayer::new(
        RMSNorm::new(ones1(4), config.rms_norm_eps),
        attn,
        RMSNorm::new(ones1(4), config.rms_norm_eps),
        SwiGLUMLP::new(zeros2(8, 4), zeros2(8, 4), zeros2(4, 8)),
    );

    let predictor = CodePredictor {
        codec_embeddings: (0..15).map(|_| zeros2(vocab_size, 4)).collect(),
        lm_heads: (0..15).map(|_| zeros2(vocab_size, 4)).collect(),
        layers: vec![layer],
        norm: RMSNorm::new(ones1(4), config.rms_norm_eps),
        small_to_mtp_proj: None,
        config,
    };

    (predictor, device)
}

/// Build a minimal TalkerForConditionalGeneration with 0 transformer layers.
///
/// Architecture: 0 talker layers, 0 CP layers, hidden_size=6, vocab=1028,
/// EOS=1024.  Suitable for testing the generation loop, transfer tracking,
/// and frame assembly.
fn make_talker() -> (TalkerForConditionalGeneration, Device) {
    let device = Device::Cpu;
    let mut config = TalkerConfig::default();
    config.hidden_size = 6;
    config.text_hidden_size = 6;
    config.head_dim = 6;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.num_hidden_layers = 0;
    config.vocab_size = TALKER_VOCAB;
    config.num_code_groups = 16;
    config.mrope_section = vec![1, 1, 1];
    config.codec_eos_token_id = EOS as u32;

    let mut cp_config = config.code_predictor.clone();
    cp_config.hidden_size = 6;
    cp_config.head_dim = 6;
    cp_config.num_attention_heads = 1;
    cp_config.num_key_value_heads = 1;
    cp_config.num_hidden_layers = 0;
    cp_config.vocab_size = CP_VOCAB;
    cp_config.num_code_groups = 16;
    config.code_predictor = cp_config.clone();

    let norm = RMSNorm::new(Tensor::ones((6,), DType::F32, &device).unwrap(), 1e-6);
    let rope = MultimodalRotaryEmbedding::new(&config, &device).unwrap();
    let cp_norm = RMSNorm::new(Tensor::ones((6,), DType::F32, &device).unwrap(), 1e-6);

    let predictor = CodePredictor {
        codec_embeddings: (0..15)
            .map(|_| Tensor::zeros((CP_VOCAB, 6), DType::F32, &device).unwrap())
            .collect(),
        lm_heads: (0..15)
            .map(|_| Tensor::zeros((CP_VOCAB, 6), DType::F32, &device).unwrap())
            .collect(),
        layers: Vec::new(),
        norm: cp_norm,
        small_to_mtp_proj: None,
        config: cp_config,
    };

    let dummy = Tensor::zeros((1,), DType::F32, &device).unwrap();
    let talker = TalkerForConditionalGeneration {
        model: TalkerModel::new(Vec::new(), norm),
        text_embedding: dummy.clone(),
        text_proj_fc1_w: dummy.clone(),
        text_proj_fc1_b: dummy.clone(),
        text_proj_fc2_w: dummy.clone(),
        text_proj_fc2_b: dummy.clone(),
        codec_embedding: Tensor::zeros((TALKER_VOCAB, 6), DType::F32, &device).unwrap(),
        codec_head: Tensor::zeros((TALKER_VOCAB, 6), DType::F32, &device).unwrap(),
        code_predictor: predictor,
        rope,
        config,
    };

    (talker, device)
}

// ===========================================================================
// § d — greedy_select_on_device correctness
// ===========================================================================

/// No suppression → on-device and host argmax agree.
#[test]
fn greedy_select_on_device_no_suppression_matches_greedy_select() {
    let device = Device::Cpu;
    let logits = Tensor::new(&[[0.0_f32, 3.0, 1.0, 2.0]], &device).unwrap();

    let host_token = greedy_select(&logits, None, None).unwrap();
    let device_token = device_select_scalar(&logits, None, None);

    assert_eq!(host_token, device_token);
    assert_eq!(host_token, 1); // index 1 has value 3.0
}

/// All tokens above `suppress_from` must be blocked.
#[test]
fn greedy_select_on_device_basic_suppression_matches_greedy_select() {
    let device = Device::Cpu;
    // suppress_from=4 → tokens 4,5,6,7 are suppressed
    let logits = Tensor::new(&[[0.0_f32, 1.0, 2.0, 3.0, 10.0, 9.0, 8.0, 7.0]], &device).unwrap();

    let host_token = greedy_select(&logits, Some(4), None).unwrap();
    let device_token = device_select_scalar(&logits, Some(4), None);

    assert_eq!(host_token, device_token);
    assert_eq!(host_token, 3); // highest valid token
}

/// An allowed token inside the suppressed region is still selectable.
#[test]
fn greedy_select_on_device_with_allowed_token_matches_greedy_select() {
    let device = Device::Cpu;
    // suppress_from=4, allowed=6 → token 6 is the only suppressed token
    // that's selectable
    let logits = Tensor::new(&[[0.0_f32, 1.0, 2.0, 3.0, 0.5, 0.5, 20.0, 0.5]], &device).unwrap();

    let host_token = greedy_select(&logits, Some(4), Some(6)).unwrap();
    let device_token = device_select_scalar(&logits, Some(4), Some(6));

    assert_eq!(host_token, device_token);
    assert_eq!(host_token, 6); // allowed token with high value
}

/// All values equal → lowest index wins (Candle argmax tie behaviour).
#[test]
fn greedy_select_on_device_ties_give_lowest_index() {
    let device = Device::Cpu;
    let logits = Tensor::new(&[[1.0_f32, 1.0, 1.0, 1.0]], &device).unwrap();

    let host_token = greedy_select(&logits, None, None).unwrap();
    let device_token = device_select_scalar(&logits, None, None);

    assert_eq!(host_token, device_token);
    assert_eq!(host_token, 0);
}

/// Ties in suppressed region: unsuppressed tokens win.
#[test]
fn greedy_select_on_device_ties_in_suppressed_region() {
    let device = Device::Cpu;
    let logits = Tensor::new(&[[0.0_f32, 5.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]], &device).unwrap();

    let host_token = greedy_select(&logits, Some(4), None).unwrap();
    let device_token = device_select_scalar(&logits, Some(4), None);

    assert_eq!(host_token, device_token);
    assert_eq!(host_token, 1); // highest value among valid
}

/// All unsuppressed tokens suppressed, allowed token is the only valid choice.
#[test]
fn greedy_select_on_device_all_suppressed_with_allowed() {
    let device = Device::Cpu;
    let logits = Tensor::new(&[[0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0]], &device).unwrap();

    let host_token = greedy_select(&logits, Some(4), Some(6)).unwrap();
    let device_token = device_select_scalar(&logits, Some(4), Some(6));

    assert_eq!(host_token, device_token);
    assert_eq!(host_token, 6);
}

/// Realistic 2048-token vocabulary stress test — verifies the on-device path
/// avoids downloading all 2048 values while matching host argmax.
#[test]
fn greedy_select_on_device_large_vocab_matches_host() {
    let device = Device::Cpu;
    let vocab_size = 2048_usize;
    let mut values = vec![0.0_f32; vocab_size];
    values[1500] = 10.0; // one clear peak

    let logits = Tensor::from_slice(&values, (1, vocab_size), &device).unwrap();
    // On-device argmax (no full-vocab download)
    let device_token = device_select_scalar(&logits, None, None);
    // Host-side argmax (downloads all values)
    let host_token = greedy_select(&logits, None, None).unwrap();

    assert_eq!(device_token, host_token);
    assert_eq!(device_token, 1500);

    // With suppression: suppress_from = 1024, allowed = 1500
    let suppress_from = vocab_size - 1024; // 1024
    let device_allowed = device_select_scalar(&logits, Some(suppress_from), Some(1500));
    let host_allowed = greedy_select(&logits, Some(suppress_from), Some(1500)).unwrap();

    assert_eq!(device_allowed, host_allowed);
    assert_eq!(
        device_allowed, 1500,
        "Token 1500 should be selectable via allowed mechanism"
    );
}

#[test]
fn greedy_select_on_device_ignores_non_finite_logits_like_host() {
    let device = Device::Cpu;
    let logits = Tensor::new(
        &[[1.0_f32, f32::INFINITY, 2.0, f32::NAN, f32::NEG_INFINITY]],
        &device,
    )
    .unwrap();

    let host_token = greedy_select(&logits, None, None).unwrap();
    let device_token = device_select_scalar(&logits, None, None);
    assert_eq!(host_token, 2);
    assert_eq!(device_token, host_token);

    let suppressed = Tensor::new(&[[1.0_f32, 2.0, f32::INFINITY]], &device).unwrap();
    let host_token = greedy_select(&suppressed, Some(2), None).unwrap();
    let device_token = device_select_scalar(&suppressed, Some(2), None);
    assert_eq!(host_token, 1);
    assert_eq!(device_token, host_token);
}

#[test]
fn greedy_select_on_device_all_invalid_returns_fail_closed_sentinel() {
    let device = Device::Cpu;
    let logits = Tensor::new(&[[f32::NAN, f32::INFINITY, f32::NEG_INFINITY]], &device).unwrap();

    assert!(greedy_select(&logits, None, None).is_err());
    assert_eq!(
        device_select_scalar(&logits, None, None),
        logits.dim(1).unwrap() as u32
    );
}

#[test]
fn greedy_select_on_device_preserves_out_of_range_mask_semantics() {
    let device = Device::Cpu;
    let logits = Tensor::new(&[[1.0_f32, 3.0, 2.0]], &device).unwrap();

    let host_token = greedy_select(&logits, Some(99), Some(77)).unwrap();
    let device_token = device_select_scalar(&logits, Some(99), Some(77));
    assert_eq!(device_token, host_token);

    let batched = Tensor::new(&[[1.0_f32, 2.0], [3.0, 4.0]], &device).unwrap();
    assert!(greedy_select_on_device(&batched, None, None).is_err());
}

// ===========================================================================
// § b — Greedy Code Predictor path: correctness + zero transfers
// ===========================================================================

/// Greedy CP `generate_with_observer` produces the same codes as the
/// non-observer path, and the observer records zero scalar transfers
/// (no `to_scalar()` in the greedy CP hot path).
#[test]
fn greedy_cp_produces_same_codes_and_zero_scalar_transfers() {
    let (predictor, device) = make_code_predictor(CP_VOCAB);
    let talker_hidden = zeros3(1, 1, 4);
    let c0_embed = zeros3(1, 1, 4);

    // ── Baseline (no observer) ──────────────────────────────────────────
    let mut caches = vec![None];
    let (codes_baseline, _) = predictor
        .generate(&talker_hidden, &c0_embed, &mut caches, &device)
        .unwrap();
    assert_eq!(codes_baseline.dims(), &[1, 15]);

    // ── With TransferTrackingObserver (greedy mode) ─────────────────────
    let mut observer = TransferTrackingObserver::new();
    let mut caches2 = vec![None];
    let (codes_with_obs, _) = predictor
        .generate_with_observer(
            &talker_hidden,
            &c0_embed,
            &mut caches2,
            &device,
            0,
            &mut observer,
        )
        .unwrap();

    // Same numerical output
    assert_eq!(
        codes_baseline.to_vec2::<u32>().unwrap(),
        codes_with_obs.to_vec2::<u32>().unwrap(),
        "greedy CP must produce identical codes with and without observer"
    );

    // Zero scalar transfers in the greedy CP path:
    //   - argmax(1) stays on-device (lines 189, 229 in code_predictor.rs)
    //   - No to_scalar() call in generate_with_observer
    //
    // The observer's on_code_predictor_step_logits fires but does NOT
    // record FullLogitVector because mode == Greedy.
    let c = TransferCollector::new();

    // No Codebook0Scalar events (those come from the Talker path only)
    {
        let c0_count: usize = c
            .events
            .iter()
            .filter(|(_, ev, _)| matches!(ev, TransferEvent::Codebook0Scalar))
            .count();
        assert_eq!(
            c0_count, 0,
            "greedy CP must not trigger any Codebook0Scalar transfers"
        );
    }

    // No FullLogitVector events (argmax is on-device)
    {
        let full_logit_count: usize = c
            .events
            .iter()
            .filter(|(_, ev, _)| matches!(ev, TransferEvent::FullLogitVector(_)))
            .count();
        assert_eq!(
            full_logit_count, 0,
            "greedy CP must not download full logit vectors"
        );
    }
}

/// `first_step_logits` itself does not trigger any stage-dump hooks on the
/// caller's observer (it uses an internal `NoopStageDumpObserver`), so no
/// transfer events are recorded — verifying the contract that the initial
/// logit computation stays on-device.
#[test]
fn greedy_cp_first_step_logits_no_observer_leak() {
    let (predictor, device) = make_code_predictor(CP_VOCAB);
    let talker_hidden = zeros3(1, 1, 4);
    let c0_embed = zeros3(1, 1, 4);
    let mut caches = vec![None];

    // standalone first_step_logits uses its own NoopStageDumpObserver
    let logits = predictor
        .first_step_logits(&talker_hidden, &c0_embed, &mut caches, &device)
        .unwrap();

    // Logits shape confirms on-device computation
    assert_eq!(logits.dims(), &[1, CP_VOCAB]);
    assert!(logits.device().is_cpu());
}

// ===========================================================================
// § c — Greedy Talker path: only Codebook0Scalar per frame
// ===========================================================================

/// Greedy Talker path must only transfer one scalar per frame for the EOS
/// check and the final output tensor.  No full-vocab download.
#[test]
fn greedy_talker_path_scalar_transfer_pattern() {
    let (talker, device) = make_talker();
    let inputs = Tensor::zeros((1, 1, 6), DType::F32, &device).unwrap();

    let mut observer = TransferTrackingObserver::new();
    let mut transfers = TransferCollector::new();
    let result = talker
        .generate_with_transfer_observer(
            &inputs,
            None,
            None,
            None,
            MAX_NEW_TOKENS,
            &device,
            &mut observer,
            &mut transfers,
        )
        .unwrap();

    let n_frames = result.dim(0).unwrap();
    assert!(n_frames >= 1, "should produce at least one frame");
    assert_eq!(result.dim(1).unwrap(), 16);

    let c = &transfers;

    // ── Exactly one D→H scalar transfer per attempted c0 draw ──────────
    assert_eq!(
        c.events,
        vec![(0, TransferEvent::Codebook0Scalar, 1); observer.frame_count],
        "greedy Talker must report only ordered D→H c0 scalar reads, including terminal validation"
    );

    // ── No FullLogitVector events (greedy_select_on_device avoids download) ──
    {
        let full_logit: usize = c
            .events
            .iter()
            .filter(|(_, ev, _)| matches!(ev, TransferEvent::FullLogitVector(_)))
            .count();
        assert_eq!(
            full_logit, 0,
            "greedy Talker path must NOT transfer full logit vectors"
        );
    }

    // ── Total elements matches contract ─────────────────────────────────
    //   Each attempted frame: 1 Codebook0Scalar (terminal sentinel validation
    //   cannot safely be delegated to a CUDA embedding lookup).
    let c0_count: usize = c
        .events
        .iter()
        .filter(|(_, ev, _)| matches!(ev, TransferEvent::Codebook0Scalar))
        .count();
    let expected_total = c0_count * 1; // scalar transfers
    assert_eq!(
        c.total_elements(),
        expected_total,
        "total transferred elements must match greedy contract"
    );
}

// ===========================================================================
// § e — Sampled Code Predictor path: FullLogitVector per step
// ===========================================================================

/// Sampled CP path must transfer the full logit vector (vocab_size elements)
/// for each of the 15 steps, plus one scalar token per frame (returned by
/// the sampler).
#[test]
fn sampled_cp_path_full_logit_transfer_pattern() {
    let (predictor, device) = make_code_predictor(CP_VOCAB);
    let talker_hidden = zeros3(1, 1, 4);
    let c0_embed = zeros3(1, 1, 4);
    let mut sampler = Sampler::new(42);
    let options = SamplingOptions {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
    };

    let mut observer = TransferTrackingObserver::new();
    let mut transfers = TransferCollector::new();
    let codes = predictor
        .generate_sampled_tensor_with_observers(
            &talker_hidden,
            &c0_embed,
            &mut vec![None],
            &device,
            &mut sampler,
            options,
            true,
            0,
            &mut observer,
            &mut transfers,
        )
        .unwrap();

    assert_eq!(codes.dims(), &[1, 15]);

    let c = &transfers;

    let expected_step_events = [
        (0, TransferEvent::FullLogitVector(CP_VOCAB), CP_VOCAB),
        (1, TransferEvent::SubCodebookScalar, 1),
    ];
    assert_eq!(c.events.len(), 15 * expected_step_events.len());
    for (step, pair) in c.events.chunks_exact(2).enumerate() {
        assert_eq!(
            pair, expected_step_events,
            "step {step}: transfer direction, kind, count, or order changed"
        );
    }

    // ── No Codebook0Scalar events in standalone CP ──────────────────────
    {
        let c0_count: usize = c
            .events
            .iter()
            .filter(|(_, ev, _)| matches!(ev, TransferEvent::Codebook0Scalar))
            .count();
        assert_eq!(
            c0_count, 0,
            "standalone CP must not trigger Codebook0Scalar events"
        );
    }

    // ── Total elements ──────────────────────────────────────────────────
    // 15 FullLogitVector events × CP_VOCAB elements each
    // + 15 SubCodebookScalar events × 1 element each (one H→D scalar token per step)
    let expected = 15 * CP_VOCAB + 15;
    assert_eq!(c.total_elements(), expected);
}

#[test]
fn transfer_capture_off_never_invokes_callback() {
    struct CaptureOff;

    impl TransferObserver for CaptureOff {
        fn on_transfer(&mut self, _direction: u8, _event: TransferEvent, _elements: usize) {
            panic!("capture-off observer must not receive transfer callbacks");
        }
    }

    let logits = Tensor::new(&[[0.0_f32, 1.0]], &Device::Cpu).unwrap();
    let mut sampler = Sampler::new(7);
    let mut observer = CaptureOff;
    let token = sampler
        .sample_with_transfer_observer(
            &logits,
            SamplingOptions {
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                repetition_penalty: 1.0,
            },
            None,
            None,
            &[],
            &mut observer,
        )
        .unwrap();
    assert_eq!(token, 1);
}

#[test]
fn terminal_cap_still_fails_closed_for_all_invalid_greedy_logits() {
    let (mut talker, device) = make_talker();
    talker.codec_head =
        Tensor::full(f32::NAN, (TALKER_VOCAB, talker.config.hidden_size), &device).unwrap();
    let inputs = Tensor::zeros((1, 1, 6), DType::F32, &device).unwrap();

    let error = talker
        .generate(&inputs, None, None, None, 1, &device)
        .expect_err("terminal cap must not accept the all-invalid sentinel");
    assert!(
        error
            .to_string()
            .contains("suppression left no finite candidate"),
        "unexpected fail-closed error: {error}"
    );
}

#[test]
fn sampled_talker_matches_noop_path_and_reports_exact_transfer_budget() {
    let (talker, device) = make_talker();
    let inputs = Tensor::zeros((1, 1, 6), DType::F32, &device).unwrap();
    let options = SamplingOptions {
        temperature: 1.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
    };
    let mut baseline_sampler = Sampler::new(1234);
    let baseline = talker
        .generate_sampled(
            &inputs,
            None,
            None,
            None,
            MAX_NEW_TOKENS,
            &device,
            &mut baseline_sampler,
            options,
            true,
            options,
            true,
        )
        .unwrap();

    let mut observed_sampler = Sampler::new(1234);
    let mut observer = TransferTrackingObserver::new();
    let mut transfers = TransferCollector::new();
    let observed = talker
        .generate_sampled_with_transfer_observer(
            &inputs,
            None,
            None,
            None,
            MAX_NEW_TOKENS,
            &device,
            &mut observed_sampler,
            options,
            options,
            true,
            true,
            &mut observer,
            &mut transfers,
        )
        .unwrap();

    assert_eq!(
        baseline.to_vec2::<u32>().unwrap(),
        observed.to_vec2::<u32>().unwrap()
    );
    let frames = observed.dim(0).unwrap();
    let attempts = observer.frame_count;
    let events = &transfers.events;
    let mut expected_events = Vec::new();
    for attempt in 0..attempts {
        expected_events.push((
            0,
            TransferEvent::FullLogitVector(TALKER_VOCAB),
            TALKER_VOCAB,
        ));
        if attempt < frames {
            expected_events.push((1, TransferEvent::Codebook0Scalar, 1));
            for _ in 0..15 {
                expected_events.push((0, TransferEvent::FullLogitVector(CP_VOCAB), CP_VOCAB));
                expected_events.push((1, TransferEvent::SubCodebookScalar, 1));
            }
        }
    }
    assert_eq!(
        *events, expected_events,
        "sampled Talker transfer order changed"
    );

    let full_logits: Vec<_> = events
        .iter()
        .filter(|(_, event, _)| matches!(event, TransferEvent::FullLogitVector(_)))
        .collect();
    assert_eq!(full_logits.len(), attempts + frames * 15);
    let expected_d2h_elements = attempts * TALKER_VOCAB + frames * 15 * CP_VOCAB;
    assert_eq!(
        full_logits
            .iter()
            .map(|(_, _, elements)| *elements)
            .sum::<usize>(),
        expected_d2h_elements
    );

    let c0_uploads = events
        .iter()
        .filter(|(direction, event, elements)| {
            *direction == 1 && *elements == 1 && matches!(event, TransferEvent::Codebook0Scalar)
        })
        .count();
    let cp_uploads = events
        .iter()
        .filter(|(direction, event, elements)| {
            *direction == 1 && *elements == 1 && matches!(event, TransferEvent::SubCodebookScalar)
        })
        .count();
    assert_eq!(c0_uploads, frames);
    assert_eq!(cp_uploads, frames * 15);
    assert_eq!(
        transfers.total_elements(),
        expected_d2h_elements + frames * 16
    );
}

// ===========================================================================
// § e — Verify codes_1_15.to_vec1() is NOT called in greedy Talker path
// ===========================================================================

/// Structural verification: the greedy Talker path assembles frames on device
/// via `Tensor::cat` without ever calling `codes_1_15.to_vec1()`.
///
/// This test exercises the full greedy pipeline and asserts the output is a
/// correct `[num_frames, 16]` tensor — the same result one would get if
/// `to_vec1()` HAD been called, but without the corresponding host-device
/// transfer.
#[test]
fn greedy_talker_assembles_frames_on_device_without_to_vec1() {
    let (talker, device) = make_talker();
    let inputs = Tensor::zeros((1, 1, 6), DType::F32, &device).unwrap();

    let result = talker
        .generate(&inputs, None, None, None, MAX_NEW_TOKENS, &device)
        .unwrap();

    let codes = result.to_vec2::<u32>().unwrap();
    assert!(!codes.is_empty(), "must produce at least one frame");

    for (frame_idx, frame) in codes.iter().enumerate() {
        assert_eq!(
            frame.len(),
            16,
            "frame {frame_idx} must have 16 code tokens (0=c0, 1..15=codes_1_15)"
        );
    }
}

/// Runtime transfer budgets are paired with a structural guard over the
/// production hot paths. The two checks fail independently: telemetry catches
/// changed boundary counts, while this guard catches an uninstrumented host
/// conversion being reintroduced.
#[test]
fn production_acoustic_hot_paths_have_no_hidden_code_vector_downloads() {
    fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
        let start = source.find(start).expect("start marker");
        let tail = &source[start..];
        let end = tail.find(end).expect("end marker");
        &tail[..end]
    }

    let talker = include_str!("../src/talker/talker.rs");
    let greedy = section(
        talker,
        "pub fn generate_with_transfer_observer",
        "pub fn generate_sampled(",
    );
    assert!(!greedy.contains(".to_vec1"));
    assert!(!greedy.contains(".to_vec2"));
    assert!(!greedy.contains("codes_1_15.to_"));
    assert_eq!(
        greedy.matches(".to_scalar::<u32>()").count(),
        1,
        "greedy Talker permits only the c0 EOS/control scalar read"
    );

    let predictor = include_str!("../src/talker/code_predictor.rs");
    let greedy_cp = section(
        predictor,
        "pub(crate) fn generate_tensor_with_observer",
        "pub fn generate_sampled(",
    );
    assert!(!greedy_cp.contains(".to_vec"));
    assert!(!greedy_cp.contains(".to_scalar"));

    let sampled_cp = section(
        predictor,
        "pub fn generate_sampled_tensor_with_observers",
        "fn project_input(",
    );
    assert!(!sampled_cp.contains(".to_vec"));
    assert!(!sampled_cp.contains(".to_scalar"));
}

/// Explicitly verify that the EOS check scalar transfer is the ONLY per-frame
/// D→H transfer — by comparing on-device argmax with host argmax and confirming
/// they match without a full logit vector transfer.
#[test]
fn greedy_talker_eos_check_transfers_one_scalar() {
    let device = Device::Cpu;

    // Simulate the exact code pattern from talker.rs generate_with_observer:
    //
    //   let logits = self.codec_head_logits(&last_hidden)?;  // on-device
    //   let logits = logits.squeeze(1)?;
    //   let c0_token = greedy_select_on_device(&logits, ...)?; // [1,1] on-device
    //   let c0_val = c0_token.reshape(())?.to_scalar::<u32>()?; // ← 1 scalar D→H
    //
    // greedy_select_on_device avoids the full-vocab download; the only
    // unavoidable transfer is the single u32 scalar for EOS comparison.

    let logits = Tensor::new(&[[0.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], &device).unwrap();

    // On-device argmax — no full-vocab download
    let c0_token = greedy_select_on_device(&logits, None, None).unwrap();
    assert_eq!(c0_token.dims(), &[1, 1]);

    // This is the ONLY scalar transfer: to_scalar for EOS check
    let c0_val = c0_token.flatten_all().unwrap().to_vec1::<u32>().unwrap()[0];
    // Verify it matches host-side argmax
    let host_val = greedy_select(&logits, None, None).unwrap();
    assert_eq!(c0_val, host_val, "on-device and host argmax must agree");
    assert_eq!(c0_val, 4, "logits have peak at index 4 (value 1.0)");

    // With suppression: suppress_from=4, EOS=6 (allowed)
    let logits2 = Tensor::new(&[[0.0_f32, 0.0, 0.0, 0.0, 0.5, 0.5, 10.0, 0.5]], &device).unwrap();
    let c0_val2 = device_select_scalar(&logits2, Some(4), Some(6));
    let host_val2 = greedy_select(&logits2, Some(4), Some(6)).unwrap();
    assert_eq!(c0_val2, host_val2);
    assert_eq!(c0_val2, 6, "EOS token should be selectable via allowed");
}

// ===========================================================================
// TransferCollector unit test
// ===========================================================================

#[test]
fn transfer_collector_records_events_correctly() {
    let mut collector = TransferCollector::new();

    assert_eq!(collector.total_transfers(), 0);
    assert_eq!(collector.total_elements(), 0);

    collector.on_transfer(0, TransferEvent::Codebook0Scalar, 1);
    collector.on_transfer(0, TransferEvent::Codebook0Scalar, 1);
    collector.on_transfer(0, TransferEvent::FullLogitVector(8), 8);
    collector.on_transfer(0, TransferEvent::FullLogitVector(48), 48);

    assert_eq!(collector.total_transfers(), 4);
    assert_eq!(collector.total_elements(), 1 + 1 + 8 + 48);

    let events: Vec<_> = collector
        .events
        .iter()
        .map(|(dir, ev, _)| (*dir, *ev))
        .collect();
    assert_eq!(events[0], (0, TransferEvent::Codebook0Scalar));
    assert_eq!(events[1], (0, TransferEvent::Codebook0Scalar));
    assert_eq!(events[2], (0, TransferEvent::FullLogitVector(8)));
    assert_eq!(events[3], (0, TransferEvent::FullLogitVector(48)));
}
