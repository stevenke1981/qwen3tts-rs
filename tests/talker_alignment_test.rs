use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};
use qwen3tts::talker::input_builder::InputBuilder;
use qwen3tts::talker::primitives::create_causal_mask;
use qwen3tts::talker::{TalkerConfig, TalkerWeightLoader};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct TextProjectionFixture {
    input_ids: Vec<Vec<u32>>,
    shape: Vec<usize>,
    output: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct CodecEmbeddingHeadFixture {
    codec_ids: Vec<Vec<u32>>,
    codec_embed_shape: Vec<usize>,
    codec_embed: Vec<f32>,
    hidden_shape: Vec<usize>,
    hidden: Vec<f32>,
    codec_logits_shape: Vec<usize>,
    codec_logits: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct AttentionLayerFixture {
    hidden_shape: Vec<usize>,
    hidden: Vec<f32>,
    position_ids_shape: Vec<usize>,
    position_ids: Vec<u32>,
    output_shape: Vec<usize>,
    output: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct ModelPrefillFixture {
    hidden_shape: Vec<usize>,
    hidden: Vec<f32>,
    output_shape: Vec<usize>,
    output: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct CodePredictorFirstStepFixture {
    talker_hidden_shape: Vec<usize>,
    talker_hidden: Vec<f32>,
    c0_token: Vec<Vec<u32>>,
    logits_shape: Vec<usize>,
    logits: Vec<f32>,
    next_token: Vec<u32>,
}

#[derive(Debug, Deserialize)]
struct CodePredictorGreedyFixture {
    talker_hidden_shape: Vec<usize>,
    talker_hidden: Vec<f32>,
    c0_token: Vec<Vec<u32>>,
    generated_shape: Vec<usize>,
    generated: Vec<Vec<u32>>,
}

#[derive(Debug, Deserialize)]
struct TalkerSingleFrameFixture {
    inputs_embeds_shape: Vec<usize>,
    inputs_embeds: Vec<f32>,
    attention_mask_shape: Vec<usize>,
    attention_mask: Vec<i64>,
    trailing_text_hidden_shape: Vec<usize>,
    trailing_text_hidden: Vec<f32>,
    tts_pad_embed_shape: Vec<usize>,
    tts_pad_embed: Vec<f32>,
    generated_shape: Vec<usize>,
    generated: Vec<Vec<u32>>,
}

#[derive(Debug, Deserialize)]
struct TalkerPromptInputBuilderFixture {
    input_ids: Vec<u32>,
    inputs_embeds_shape: Vec<usize>,
    inputs_embeds: Vec<f32>,
    attention_mask_shape: Vec<usize>,
    attention_mask: Vec<i64>,
    trailing_text_hidden_shape: Vec<usize>,
    trailing_text_hidden: Vec<f32>,
    tts_pad_embed_shape: Vec<usize>,
    tts_pad_embed: Vec<f32>,
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let x = x as f64;
        let y = y as f64;
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt() + 1e-12)
}

fn default_model_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("QWEN3_TTS_MODEL_SAFETENSORS") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }

    let home = std::env::var("USERPROFILE").ok().map(PathBuf::from)?;
    let snapshots = home
        .join(".cache")
        .join("huggingface")
        .join("hub")
        .join("models--Qwen--Qwen3-TTS-12Hz-0.6B-Base")
        .join("snapshots");
    let entries = std::fs::read_dir(snapshots).ok()?;
    for entry in entries.flatten() {
        let candidate = entry.path().join("model.safetensors");
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

fn load_text_projection_fixture(path: &Path) -> TextProjectionFixture {
    let data = std::fs::read_to_string(path).expect("read fixture");
    serde_json::from_str(&data).expect("parse fixture")
}

fn load_codec_embedding_head_fixture(path: &Path) -> CodecEmbeddingHeadFixture {
    let data = std::fs::read_to_string(path).expect("read fixture");
    serde_json::from_str(&data).expect("parse fixture")
}

fn load_attention_layer_fixture(path: &Path) -> AttentionLayerFixture {
    let data = std::fs::read_to_string(path).expect("read fixture");
    serde_json::from_str(&data).expect("parse fixture")
}

fn load_decoder_layer_fixture(path: &Path) -> AttentionLayerFixture {
    let data = std::fs::read_to_string(path).expect("read fixture");
    serde_json::from_str(&data).expect("parse fixture")
}

fn load_model_prefill_fixture(path: &Path) -> ModelPrefillFixture {
    let data = std::fs::read_to_string(path).expect("read fixture");
    serde_json::from_str(&data).expect("parse fixture")
}

fn load_code_predictor_first_step_fixture(path: &Path) -> CodePredictorFirstStepFixture {
    let data = std::fs::read_to_string(path).expect("read fixture");
    serde_json::from_str(&data).expect("parse fixture")
}

fn load_code_predictor_greedy_fixture(path: &Path) -> CodePredictorGreedyFixture {
    let data = std::fs::read_to_string(path).expect("read fixture");
    serde_json::from_str(&data).expect("parse fixture")
}

fn load_talker_single_frame_fixture(path: &Path) -> TalkerSingleFrameFixture {
    let data = std::fs::read_to_string(path).expect("read fixture");
    serde_json::from_str(&data).expect("parse fixture")
}

fn load_talker_prompt_input_builder_fixture(path: &Path) -> TalkerPromptInputBuilderFixture {
    let data = std::fs::read_to_string(path).expect("read fixture");
    serde_json::from_str(&data).expect("parse fixture")
}

fn load_talker(device: &Device) -> Option<qwen3tts::talker::TalkerForConditionalGeneration> {
    let Some(model_path) = default_model_path() else {
        eprintln!("Skipping: Qwen3-TTS model.safetensors not found");
        return None;
    };

    let loader = TalkerWeightLoader::from_safetensors(model_path, device).expect("load talker");
    Some(
        loader
            .build_talker(&TalkerConfig::default())
            .expect("build talker"),
    )
}

#[test]
#[ignore = "loads the full 0.6B talker model; run explicitly for PyTorch alignment"]
fn talker_text_projection_matches_pytorch_fixture() {
    let fixture_path = Path::new("tests/fixtures/talker_text_projection.json");
    if !fixture_path.exists() {
        eprintln!("Skipping: fixture not found at {fixture_path:?}");
        return;
    }

    let fixture = load_text_projection_fixture(fixture_path);
    assert_eq!(fixture.shape, vec![1, fixture.input_ids[0].len(), 1024]);

    let device = Device::Cpu;
    let Some(talker) = load_talker(&device) else {
        return;
    };

    let flat_ids: Vec<u32> = fixture.input_ids.iter().flatten().copied().collect();
    let input = Tensor::from_slice(
        &flat_ids,
        (fixture.input_ids.len(), fixture.input_ids[0].len()),
        &device,
    )
    .expect("input tensor");
    let output = talker.embed_text(&input).expect("embed text");
    assert_eq!(output.dims(), fixture.shape.as_slice());

    let rust = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let cos = cosine_sim(&rust, &fixture.output);
    let max_abs = rust
        .iter()
        .zip(&fixture.output)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("talker text projection cosine_vs_pt={cos:.8} max_abs={max_abs:.8}");
    assert!(cos >= 0.999, "cosine {cos:.8} < 0.999");
    assert!(max_abs <= 1e-3, "max_abs {max_abs:.8} > 1e-3");
}

#[test]
#[ignore = "loads the full 0.6B talker model; run explicitly for PyTorch alignment"]
fn talker_codec_embedding_and_head_match_pytorch_fixture() {
    let fixture_path = Path::new("tests/fixtures/talker_codec_embedding_head.json");
    if !fixture_path.exists() {
        eprintln!("Skipping: fixture not found at {fixture_path:?}");
        return;
    }

    let fixture = load_codec_embedding_head_fixture(fixture_path);
    let device = Device::Cpu;
    let Some(talker) = load_talker(&device) else {
        return;
    };

    let flat_ids: Vec<u32> = fixture.codec_ids.iter().flatten().copied().collect();
    let codec_ids = Tensor::from_slice(
        &flat_ids,
        (fixture.codec_ids.len(), fixture.codec_ids[0].len()),
        &device,
    )
    .expect("codec ids");
    let codec_embed = talker.embed_codec(&codec_ids).expect("codec embed");
    assert_eq!(codec_embed.dims(), fixture.codec_embed_shape.as_slice());
    let rust_embed = codec_embed.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let embed_cos = cosine_sim(&rust_embed, &fixture.codec_embed);
    let embed_max_abs = rust_embed
        .iter()
        .zip(&fixture.codec_embed)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let hidden = Tensor::from_slice(
        &fixture.hidden,
        (
            fixture.hidden_shape[0],
            fixture.hidden_shape[1],
            fixture.hidden_shape[2],
        ),
        &device,
    )
    .expect("hidden");
    let logits = talker.codec_head_logits(&hidden).expect("codec logits");
    assert_eq!(logits.dims(), fixture.codec_logits_shape.as_slice());
    let rust_logits = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let logits_cos = cosine_sim(&rust_logits, &fixture.codec_logits);
    let logits_max_abs = rust_logits
        .iter()
        .zip(&fixture.codec_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("talker codec embedding cosine_vs_pt={embed_cos:.8} max_abs={embed_max_abs:.8}");
    println!("talker codec head cosine_vs_pt={logits_cos:.8} max_abs={logits_max_abs:.8}");
    assert!(
        embed_cos >= 0.999,
        "embedding cosine {embed_cos:.8} < 0.999"
    );
    assert!(
        embed_max_abs <= 1e-6,
        "embedding max_abs {embed_max_abs:.8} > 1e-6"
    );
    assert!(logits_cos >= 0.999, "logits cosine {logits_cos:.8} < 0.999");
    assert!(
        logits_max_abs <= 1e-3,
        "logits max_abs {logits_max_abs:.8} > 1e-3"
    );
}

#[test]
#[ignore = "loads the full 0.6B talker model; run explicitly for PyTorch alignment"]
fn talker_attention_layer0_matches_pytorch_fixture() {
    let fixture_path = Path::new("tests/fixtures/talker_attention_layer0.json");
    if !fixture_path.exists() {
        eprintln!("Skipping: fixture not found at {fixture_path:?}");
        return;
    }

    let fixture = load_attention_layer_fixture(fixture_path);
    let device = Device::Cpu;
    let Some(talker) = load_talker(&device) else {
        return;
    };

    let hidden = Tensor::from_slice(
        &fixture.hidden,
        (
            fixture.hidden_shape[0],
            fixture.hidden_shape[1],
            fixture.hidden_shape[2],
        ),
        &device,
    )
    .expect("hidden");
    let position_ids = Tensor::from_slice(
        &fixture.position_ids,
        (
            fixture.position_ids_shape[0],
            fixture.position_ids_shape[1],
            fixture.position_ids_shape[2],
        ),
        &device,
    )
    .expect("position ids");
    let (cos, sin) = talker.rope.forward(&hidden, &position_ids).expect("rope");
    let mask = create_causal_mask(fixture.hidden_shape[1], &device).expect("mask");
    let (output, _cache) = talker.model.layers[0]
        .self_attn
        .forward(&hidden, &cos, &sin, Some(&mask), None)
        .expect("attention forward");
    assert_eq!(output.dims(), fixture.output_shape.as_slice());

    let rust = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let cos = cosine_sim(&rust, &fixture.output);
    let max_abs = rust
        .iter()
        .zip(&fixture.output)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("talker attention layer0 cosine_vs_pt={cos:.8} max_abs={max_abs:.8}");
    assert!(cos >= 0.999, "attention cosine {cos:.8} < 0.999");
    assert!(max_abs <= 1e-3, "attention max_abs {max_abs:.8} > 1e-3");
}

#[test]
#[ignore = "loads the full 0.6B talker model; run explicitly for PyTorch alignment"]
fn talker_decoder_layer0_matches_pytorch_fixture() {
    let fixture_path = Path::new("tests/fixtures/talker_decoder_layer0.json");
    if !fixture_path.exists() {
        eprintln!("Skipping: fixture not found at {fixture_path:?}");
        return;
    }

    let fixture = load_decoder_layer_fixture(fixture_path);
    let device = Device::Cpu;
    let Some(talker) = load_talker(&device) else {
        return;
    };

    let hidden = Tensor::from_slice(
        &fixture.hidden,
        (
            fixture.hidden_shape[0],
            fixture.hidden_shape[1],
            fixture.hidden_shape[2],
        ),
        &device,
    )
    .expect("hidden");
    let position_ids = Tensor::from_slice(
        &fixture.position_ids,
        (
            fixture.position_ids_shape[0],
            fixture.position_ids_shape[1],
            fixture.position_ids_shape[2],
        ),
        &device,
    )
    .expect("position ids");
    let (cos, sin) = talker.rope.forward(&hidden, &position_ids).expect("rope");
    let mask = create_causal_mask(fixture.hidden_shape[1], &device).expect("mask");
    let (output, _cache) = talker.model.layers[0]
        .forward(&hidden, &cos, &sin, Some(&mask), None)
        .expect("decoder layer forward");
    assert_eq!(output.dims(), fixture.output_shape.as_slice());

    let rust = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let cos = cosine_sim(&rust, &fixture.output);
    let max_abs = rust
        .iter()
        .zip(&fixture.output)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("talker decoder layer0 cosine_vs_pt={cos:.8} max_abs={max_abs:.8}");
    assert!(cos >= 0.999, "decoder layer cosine {cos:.8} < 0.999");
    assert!(max_abs <= 1e-3, "decoder layer max_abs {max_abs:.8} > 1e-3");
}

#[test]
#[ignore = "loads the full 0.6B talker model; run explicitly for PyTorch alignment"]
fn talker_model_prefill_matches_pytorch_fixture() {
    let fixture_path = Path::new("tests/fixtures/talker_model_prefill.json");
    if !fixture_path.exists() {
        eprintln!("Skipping: fixture not found at {fixture_path:?}");
        return;
    }

    let fixture = load_model_prefill_fixture(fixture_path);
    let device = Device::Cpu;
    let Some(talker) = load_talker(&device) else {
        return;
    };

    let hidden = Tensor::from_slice(
        &fixture.hidden,
        (
            fixture.hidden_shape[0],
            fixture.hidden_shape[1],
            fixture.hidden_shape[2],
        ),
        &device,
    )
    .expect("hidden");
    let seq_len = fixture.hidden_shape[1];
    let pos: Vec<u32> = (0..3)
        .flat_map(|_| (0..seq_len).map(|v| v as u32))
        .collect();
    let position_ids = Tensor::from_slice(&pos, (3, 1, seq_len), &device).expect("position ids");
    let (cos, sin) = talker.rope.forward(&hidden, &position_ids).expect("rope");
    let mask = create_causal_mask(seq_len, &device).expect("mask");
    let mut caches = vec![None; talker.config.num_hidden_layers];
    let (output, _cache) = talker
        .model
        .forward(&hidden, &cos, &sin, Some(&mask), &mut caches)
        .expect("model forward");
    assert_eq!(output.dims(), fixture.output_shape.as_slice());

    let rust = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let cos = cosine_sim(&rust, &fixture.output);
    let max_abs = rust
        .iter()
        .zip(&fixture.output)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("talker model prefill cosine_vs_pt={cos:.8} max_abs={max_abs:.8}");
    assert!(cos >= 0.999, "model prefill cosine {cos:.8} < 0.999");
    assert!(max_abs <= 1e-3, "model prefill max_abs {max_abs:.8} > 1e-3");
}

#[test]
#[ignore = "loads the full 0.6B talker model; run explicitly for PyTorch alignment"]
fn code_predictor_first_step_matches_pytorch_fixture() {
    let fixture_path = Path::new("tests/fixtures/code_predictor_first_step.json");
    if !fixture_path.exists() {
        eprintln!("Skipping: fixture not found at {fixture_path:?}");
        return;
    }

    let fixture = load_code_predictor_first_step_fixture(fixture_path);
    let device = Device::Cpu;
    let Some(talker) = load_talker(&device) else {
        return;
    };

    let talker_hidden = Tensor::from_slice(
        &fixture.talker_hidden,
        (
            fixture.talker_hidden_shape[0],
            fixture.talker_hidden_shape[1],
            fixture.talker_hidden_shape[2],
        ),
        &device,
    )
    .expect("talker hidden");
    let c0_flat: Vec<u32> = fixture.c0_token.iter().flatten().copied().collect();
    let c0 = Tensor::from_slice(
        &c0_flat,
        (fixture.c0_token.len(), fixture.c0_token[0].len()),
        &device,
    )
    .expect("c0 token");
    let c0_embed = talker.embed_codec(&c0).expect("c0 embed");
    let mut caches = vec![None; talker.config.code_predictor.num_hidden_layers];
    let logits = talker
        .code_predictor
        .first_step_logits(&talker_hidden, &c0_embed, &mut caches, &device)
        .expect("first step logits");
    assert_eq!(logits.dims(), fixture.logits_shape.as_slice());

    let rust = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let cos = cosine_sim(&rust, &fixture.logits);
    let max_abs = rust
        .iter()
        .zip(&fixture.logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let next = logits.argmax(1).unwrap().to_vec1::<u32>().unwrap();

    println!("code predictor first-step cosine_vs_pt={cos:.8} max_abs={max_abs:.8} next={next:?}");
    assert_eq!(next, fixture.next_token);
    assert!(cos >= 0.999, "code predictor cosine {cos:.8} < 0.999");
    assert!(
        max_abs <= 1e-3,
        "code predictor max_abs {max_abs:.8} > 1e-3"
    );
}

#[test]
#[ignore = "loads the full 0.6B talker model; run explicitly for PyTorch alignment"]
fn code_predictor_greedy_matches_pytorch_fixture() {
    let fixture_path = Path::new("tests/fixtures/code_predictor_greedy.json");
    if !fixture_path.exists() {
        eprintln!("Skipping: fixture not found at {fixture_path:?}");
        return;
    }

    let fixture = load_code_predictor_greedy_fixture(fixture_path);
    let device = Device::Cpu;
    let Some(talker) = load_talker(&device) else {
        return;
    };

    let talker_hidden = Tensor::from_slice(
        &fixture.talker_hidden,
        (
            fixture.talker_hidden_shape[0],
            fixture.talker_hidden_shape[1],
            fixture.talker_hidden_shape[2],
        ),
        &device,
    )
    .expect("talker hidden");
    let c0_flat: Vec<u32> = fixture.c0_token.iter().flatten().copied().collect();
    let c0 = Tensor::from_slice(
        &c0_flat,
        (fixture.c0_token.len(), fixture.c0_token[0].len()),
        &device,
    )
    .expect("c0 token");
    let c0_embed = talker.embed_codec(&c0).expect("c0 embed");
    let mut caches = vec![None; talker.config.code_predictor.num_hidden_layers];
    let (codes, _updated_caches) = talker
        .code_predictor
        .generate(&talker_hidden, &c0_embed, &mut caches, &device)
        .expect("code predictor greedy generate");
    assert_eq!(codes.dims(), fixture.generated_shape.as_slice());

    let rust = codes.to_vec2::<u32>().expect("generated token ids");
    println!(
        "code predictor greedy rust={rust:?} pytorch={:?}",
        fixture.generated
    );
    assert_eq!(rust, fixture.generated);
}

#[test]
#[ignore = "loads the full 0.6B talker model; run explicitly for PyTorch alignment"]
fn talker_single_frame_matches_pytorch_fixture() {
    let fixture_path = Path::new("tests/fixtures/talker_single_frame.json");
    if !fixture_path.exists() {
        eprintln!("Skipping: fixture not found at {fixture_path:?}");
        return;
    }

    let fixture = load_talker_single_frame_fixture(fixture_path);
    let device = Device::Cpu;
    let Some(talker) = load_talker(&device) else {
        return;
    };

    let inputs_embeds = Tensor::from_slice(
        &fixture.inputs_embeds,
        (
            fixture.inputs_embeds_shape[0],
            fixture.inputs_embeds_shape[1],
            fixture.inputs_embeds_shape[2],
        ),
        &device,
    )
    .expect("inputs embeds");
    let attention_mask = Tensor::from_slice(
        &fixture.attention_mask,
        (
            fixture.attention_mask_shape[0],
            fixture.attention_mask_shape[1],
        ),
        &device,
    )
    .expect("attention mask");
    let trailing_text_hidden = Tensor::from_slice(
        &fixture.trailing_text_hidden,
        (
            fixture.trailing_text_hidden_shape[0],
            fixture.trailing_text_hidden_shape[1],
            fixture.trailing_text_hidden_shape[2],
        ),
        &device,
    )
    .expect("trailing text hidden");
    let tts_pad_embed = Tensor::from_slice(
        &fixture.tts_pad_embed,
        (
            fixture.tts_pad_embed_shape[0],
            fixture.tts_pad_embed_shape[1],
            fixture.tts_pad_embed_shape[2],
        ),
        &device,
    )
    .expect("tts pad embed");

    let codes = talker
        .generate(
            &inputs_embeds,
            Some(&attention_mask),
            Some(&trailing_text_hidden),
            Some(&tts_pad_embed),
            1,
            &device,
        )
        .expect("talker single-frame generate");
    assert_eq!(codes.dims(), fixture.generated_shape.as_slice());

    let rust = codes.to_vec2::<u32>().expect("generated token ids");
    println!(
        "talker single-frame rust={rust:?} pytorch={:?}",
        fixture.generated
    );
    assert_eq!(rust, fixture.generated);
}

#[test]
#[ignore = "loads the full 0.6B talker model; run explicitly for PyTorch alignment"]
fn talker_two_frame_autoregressive_matches_pytorch_fixture() {
    let fixture_path = Path::new("tests/fixtures/talker_two_frame.json");
    if !fixture_path.exists() {
        eprintln!("Skipping: fixture not found at {fixture_path:?}");
        return;
    }

    let fixture = load_talker_single_frame_fixture(fixture_path);
    let device = Device::Cpu;
    let Some(talker) = load_talker(&device) else {
        return;
    };

    let inputs_embeds = Tensor::from_slice(
        &fixture.inputs_embeds,
        (
            fixture.inputs_embeds_shape[0],
            fixture.inputs_embeds_shape[1],
            fixture.inputs_embeds_shape[2],
        ),
        &device,
    )
    .expect("inputs embeds");
    let attention_mask = Tensor::from_slice(
        &fixture.attention_mask,
        (
            fixture.attention_mask_shape[0],
            fixture.attention_mask_shape[1],
        ),
        &device,
    )
    .expect("attention mask");
    let trailing_text_hidden = Tensor::from_slice(
        &fixture.trailing_text_hidden,
        (
            fixture.trailing_text_hidden_shape[0],
            fixture.trailing_text_hidden_shape[1],
            fixture.trailing_text_hidden_shape[2],
        ),
        &device,
    )
    .expect("trailing text hidden");
    let tts_pad_embed = Tensor::from_slice(
        &fixture.tts_pad_embed,
        (
            fixture.tts_pad_embed_shape[0],
            fixture.tts_pad_embed_shape[1],
            fixture.tts_pad_embed_shape[2],
        ),
        &device,
    )
    .expect("tts pad embed");

    let codes = talker
        .generate(
            &inputs_embeds,
            Some(&attention_mask),
            Some(&trailing_text_hidden),
            Some(&tts_pad_embed),
            2,
            &device,
        )
        .expect("talker two-frame generate");
    assert_eq!(codes.dims(), fixture.generated_shape.as_slice());

    let rust = codes.to_vec2::<u32>().expect("generated token ids");
    println!(
        "talker two-frame rust={rust:?} pytorch={:?}",
        fixture.generated
    );
    assert_eq!(rust, fixture.generated);
}

#[test]
#[ignore = "loads the full 0.6B talker model; run explicitly for PyTorch alignment"]
fn talker_prompt_input_builder_matches_pytorch_fixture() {
    let fixture_path = Path::new("tests/fixtures/talker_prompt_input_builder.json");
    if !fixture_path.exists() {
        eprintln!("Skipping: fixture not found at {fixture_path:?}");
        return;
    }

    let fixture = load_talker_prompt_input_builder_fixture(fixture_path);
    let device = Device::Cpu;
    let Some(talker) = load_talker(&device) else {
        return;
    };

    let builder = InputBuilder::new(&talker, &device);
    let (inputs_embeds, attention_mask, trailing_text_hidden, tts_pad_embed) = builder
        .build(&fixture.input_ids, "Chinese", None)
        .expect("build prompt inputs");

    assert_eq!(inputs_embeds.dims(), fixture.inputs_embeds_shape.as_slice());
    assert_eq!(
        attention_mask.dims(),
        fixture.attention_mask_shape.as_slice()
    );
    assert_eq!(
        trailing_text_hidden.dims(),
        fixture.trailing_text_hidden_shape.as_slice()
    );
    assert_eq!(tts_pad_embed.dims(), fixture.tts_pad_embed_shape.as_slice());

    let rust_inputs = inputs_embeds
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let inputs_cos = cosine_sim(&rust_inputs, &fixture.inputs_embeds);
    let inputs_max_abs = rust_inputs
        .iter()
        .zip(&fixture.inputs_embeds)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let rust_mask = attention_mask
        .flatten_all()
        .unwrap()
        .to_vec1::<i64>()
        .unwrap();

    let rust_trailing = trailing_text_hidden
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let trailing_cos = cosine_sim(&rust_trailing, &fixture.trailing_text_hidden);
    let trailing_max_abs = rust_trailing
        .iter()
        .zip(&fixture.trailing_text_hidden)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let rust_pad = tts_pad_embed
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let pad_cos = cosine_sim(&rust_pad, &fixture.tts_pad_embed);
    let pad_max_abs = rust_pad
        .iter()
        .zip(&fixture.tts_pad_embed)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!(
        "talker prompt input-builder inputs cosine_vs_pt={inputs_cos:.8} max_abs={inputs_max_abs:.8}"
    );
    println!(
        "talker prompt input-builder trailing cosine_vs_pt={trailing_cos:.8} max_abs={trailing_max_abs:.8}"
    );
    println!("talker prompt input-builder pad cosine_vs_pt={pad_cos:.8} max_abs={pad_max_abs:.8}");

    assert_eq!(rust_mask, fixture.attention_mask);
    assert!(inputs_cos >= 0.999, "inputs cosine {inputs_cos:.8} < 0.999");
    assert!(
        inputs_max_abs <= 1e-3,
        "inputs max_abs {inputs_max_abs:.8} > 1e-3"
    );
    assert!(
        trailing_cos >= 0.999,
        "trailing cosine {trailing_cos:.8} < 0.999"
    );
    assert!(
        trailing_max_abs <= 1e-3,
        "trailing max_abs {trailing_max_abs:.8} > 1e-3"
    );
    assert!(pad_cos >= 0.999, "pad cosine {pad_cos:.8} < 0.999");
    assert!(pad_max_abs <= 1e-3, "pad max_abs {pad_max_abs:.8} > 1e-3");
}
