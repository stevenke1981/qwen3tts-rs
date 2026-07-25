use std::collections::HashMap;
use std::path::{Path, PathBuf};

use qwen3tts::text_frontend::prompt_templates::{
    build_assistant_prompt, build_instruction_prompt, build_reference_prompt,
    reference_text_tokens_from_prompt_ids,
};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

#[derive(Debug, Deserialize)]
struct PromptIdMatrixFixture {
    version: u32,
    models: Vec<ModelFixture>,
    samples: PromptSamples,
}

#[derive(Debug, Deserialize)]
struct PromptSamples {
    main_text: String,
    reference_text: String,
    instruction_text: String,
}

#[derive(Debug, Deserialize)]
struct ModelFixture {
    model_id: String,
    model_revision: String,
    config_sha256: String,
    supports_voice_clone: bool,
    supports_voice_design: bool,
    supports_speaker_presets: bool,
    expected_speaker: Option<ExpectedSpeaker>,
    cases: Vec<PromptCase>,
}

#[derive(Debug, Deserialize, PartialEq)]
struct ExpectedSpeaker {
    name: String,
    token_id: u32,
}

#[derive(Debug, Deserialize)]
struct PromptCase {
    case: String,
    x_vector_only: bool,
    main_prompt_ids: Vec<u32>,
    reference_prompt_ids: Vec<u32>,
    reference_body_ids: Vec<u32>,
    instruction_prompt_ids: Vec<u32>,
}

#[derive(Debug, Deserialize)]
struct ModelMatrixConfig {
    models: Vec<ModelMatrixEntry>,
}

#[derive(Debug, Deserialize)]
struct ModelMatrixEntry {
    id: String,
    voice_clone: bool,
    custom_voice: bool,
    voice_design: bool,
}

#[test]
#[ignore = "requires local HF model/tokenizer artifacts; run with --ignored"]
fn prompt_id_matrix_matches_official_oracle() {
    let fixture_path = Path::new("fixtures/alignment/p01_prompt_id_matrix.json");
    if !fixture_path.exists() {
        panic!("FIXTURE_MISSING: {}", fixture_path.display());
    }

    let fixture_data = std::fs::read_to_string(fixture_path)
        .unwrap_or_else(|err| panic!("FIXTURE_MISSING: read fixture failed: {err}"));
    let fixture: PromptIdMatrixFixture =
        serde_json::from_str(&fixture_data).expect("parse p01_prompt_id_matrix fixture");

    assert_eq!(fixture.version, 1);
    assert_eq!(
        fixture.models.len(),
        5,
        "expected 5 model entries in prompt matrix"
    );
    assert_eq!(
        fixture.samples.main_text,
        "請你用自然語調，讀一段簡短的中文介紹。"
    );
    assert_eq!(
        fixture.samples.reference_text,
        "參考文字：這段句子用來建立語者風格與語感。"
    );
    assert_eq!(fixture.samples.instruction_text, "請用更溫和、清晰的語氣");

    let model_dir = std::env::var("QWEN3_TTS_REAL_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            panic!("FIXTURE_MISSING: missing QWEN3_TTS_REAL_MODEL_DIR environment variable")
        });
    if !model_dir.exists() {
        panic!(
            "FIXTURE_MISSING: model dir does not exist: {}",
            model_dir.display()
        );
    }

    let config_path = model_dir.join("config.json");
    if !config_path.exists() {
        panic!(
            "FIXTURE_MISSING: model config.json missing: {}",
            config_path.display()
        );
    }

    let tokenizer_path = {
        let direct = model_dir.join("tokenizer.json");
        let speech = model_dir.join("speech_tokenizer").join("tokenizer.json");
        if direct.exists() {
            direct
        } else if speech.exists() {
            speech
        } else {
            panic!(
                "FIXTURE_MISSING: no tokenizer.json in {}",
                model_dir.display()
            );
        }
    };

    let metadata = load_metadata_from_config(&config_path).expect("load 0.6B base config metadata");
    let tokenizer_path = tokenizer_path.as_path().to_str().unwrap_or_else(|| {
        panic!(
            "FIXTURE_MISSING: model tokenizer path is non-utf8: {}",
            tokenizer_path.display()
        )
    });
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .unwrap_or_else(|err| panic!("FIXTURE_MISSING: failed to load tokenizer: {err}"));

    let expected_matrix = load_expected_matrix();
    let mut seen_model_ids = HashMap::<String, ()>::new();

    let main_expected = encode_ids(
        &tokenizer,
        &build_assistant_prompt(&fixture.samples.main_text),
    );
    let reference_expected = encode_ids(
        &tokenizer,
        &build_reference_prompt(&fixture.samples.reference_text),
    );
    let instruction_expected = encode_ids(
        &tokenizer,
        &build_instruction_prompt(&fixture.samples.instruction_text),
    );
    let reference_body_expected = reference_text_tokens_from_prompt_ids(&reference_expected)
        .unwrap_or_else(|err| {
            panic!(
                "reference_text_tokens_from_prompt_ids failed for fixture reference prompt: {err}"
            )
        });

    for model in &fixture.models {
        let has_seen = seen_model_ids.insert(model.model_id.clone(), ());
        assert!(
            has_seen.is_none(),
            "duplicate model_id in fixture: {}",
            model.model_id
        );

        let expected = expected_matrix
            .get(&model.model_id)
            .unwrap_or_else(|| panic!("missing model matrix entry for {}", model.model_id));

        let (expected_cases, expect_speaker, expect_model_size, expect_instruction) =
            expected_case_profile(&model.model_id);
        assert_eq!(
            model.supports_voice_clone, expected.voice_clone,
            "{} supports_voice_clone mismatch",
            model.model_id
        );
        assert_eq!(
            model.supports_speaker_presets, expected.custom_voice,
            "{} supports_speaker_presets mismatch",
            model.model_id
        );
        assert_eq!(
            model.supports_voice_design, expected.voice_design,
            "{} supports_voice_design mismatch",
            model.model_id
        );
        assert_eq!(
            model.cases.len(),
            expected_cases.len(),
            "{} unexpected number of cases",
            model.model_id
        );
        assert_eq!(
            model.config_sha256,
            expected_config_sha(&model.model_id),
            "{} config hash changed from oracle",
            model.model_id
        );

        for (idx, case) in model.cases.iter().enumerate() {
            let expected_case = expected_cases[idx];
            assert_eq!(
                case.case, expected_case,
                "{} case name mismatch",
                model.model_id
            );
            assert_eq!(
                case.main_prompt_ids, main_expected,
                "{} main prompt ids changed",
                model.model_id
            );

            let is_icl = expected_case == "icl";
            if is_icl {
                assert_eq!(
                    case.reference_prompt_ids, reference_expected,
                    "{} reference prompt ids mismatch",
                    model.model_id
                );
                assert_eq!(
                    case.reference_body_ids, reference_body_expected,
                    "{} reference body ids mismatch",
                    model.model_id
                );
            } else {
                assert!(
                    case.reference_prompt_ids.is_empty(),
                    "{} should not include reference ids",
                    model.model_id
                );
                assert!(
                    case.reference_body_ids.is_empty(),
                    "{} should not include reference body ids",
                    model.model_id
                );
            }

            if expected_case == "x_vector_only" {
                assert!(
                    case.x_vector_only,
                    "{} x_vector_only flag mismatch",
                    model.model_id
                );
            } else {
                assert!(
                    !case.x_vector_only,
                    "{} x_vector_only flag mismatch",
                    model.model_id
                );
            }

            if expect_instruction {
                assert_eq!(
                    case.instruction_prompt_ids, instruction_expected,
                    "{} instruction ids mismatch",
                    model.model_id
                );
            } else {
                assert!(
                    case.instruction_prompt_ids.is_empty(),
                    "{} unexpected instruction ids",
                    model.model_id
                );
            }
        }

        match model.model_id.as_str() {
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base" => {
                assert_eq!(model.model_revision, expected_model_revision("0.6B-Base"));
                assert!(model.expected_speaker.is_none());
                assert_eq!(expect_model_size, "0b6");
            }
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice" => {
                assert_eq!(
                    model.model_revision,
                    expected_model_revision("0.6B-CustomVoice")
                );
                assert_eq!(expect_model_size, "0b6");
            }
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base" => {
                assert_eq!(model.model_revision, expected_model_revision("1.7B-Base"));
                assert_eq!(expect_model_size, "1b7");
            }
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" => {
                assert_eq!(
                    model.model_revision,
                    expected_model_revision("1.7B-CustomVoice")
                );
                assert_eq!(expect_model_size, "1b7");
            }
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign" => {
                assert_eq!(
                    model.model_revision,
                    expected_model_revision("1.7B-VoiceDesign")
                );
                assert_eq!(expect_model_size, "1b7");
            }
            _ => panic!("unexpected model_id in fixture: {}", model.model_id),
        }

        if expect_speaker {
            let expected = model
                .expected_speaker
                .as_ref()
                .unwrap_or_else(|| panic!("expected speaker missing for {}", model.model_id));
            assert!(
                !expected.name.trim().is_empty(),
                "{} speaker name is empty",
                model.model_id
            );
            assert_eq!(
                expected.token_id, 2878,
                "{} speaker token changed",
                model.model_id
            );
            assert_eq!(
                expected.name.to_lowercase(),
                "dylan",
                "{} expected speaker name is not dylan",
                model.model_id
            );
        } else {
            assert!(
                model.expected_speaker.is_none(),
                "{} unexpected speaker entry",
                model.model_id
            );
        }
    }

    // Literal metadata checks against the real 0.6B Base fixture entry.
    let base_06_entry = fixture
        .models
        .iter()
        .find(|model| model.model_id == "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
        .expect("missing 0.6B-Base fixture entry");

    assert_eq!(metadata.tts_model_size, "0b6");
    assert_eq!(metadata.tts_model_type, "base");
    assert_eq!(base_06_entry.supports_voice_clone, true);
    assert_eq!(base_06_entry.supports_voice_design, false);
    assert_eq!(base_06_entry.supports_speaker_presets, false);

    let snapshot_revision = model_revision_from_path(&model_dir).unwrap_or_else(|| {
        panic!("FIXTURE_MISSING: no snapshot revision under QWEN3_TTS_REAL_MODEL_DIR")
    });
    assert_eq!(
        snapshot_revision, base_06_entry.model_revision,
        "base model revision mismatch"
    );
    assert_eq!(
        sha256_file(&config_path),
        base_06_entry.config_sha256,
        "base model config hash mismatch"
    );
}

fn expected_case_profile(model_id: &str) -> (&'static [&'static str], bool, &'static str, bool) {
    match model_id {
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base" => (&["x_vector_only", "icl"], false, "0b6", false),
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base" => (&["x_vector_only", "icl"], false, "1b7", false),
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice" => (&["custom_voice"], true, "0b6", false),
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" => (&["custom_voice"], true, "1b7", true),
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign" => (&["voice_design"], false, "1b7", true),
        _ => panic!("unexpected model_id in fixture: {model_id}"),
    }
}

fn expected_model_revision(model_id: &str) -> &'static str {
    match model_id {
        "0.6B-Base" => "5d83992436eae1d760afd27aff78a71d676296fc",
        "0.6B-CustomVoice" => "85e237c12c027371202489a0ec509ded67b5e4b5",
        "1.7B-Base" => "fd4b254389122332181a7c3db7f27e918eec64e3",
        "1.7B-CustomVoice" => "0c0e3051f131929182e2c023b9537f8b1c68adfe",
        "1.7B-VoiceDesign" => "5ecdb67327fd37bb2e042aab12ff7391903235d3",
        _ => panic!("unknown model label: {model_id}"),
    }
}

fn expected_config_sha(model_id: &str) -> &'static str {
    match model_id {
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base" => {
            "2e714c787c8edb98b05432685cddb634add2de4d4e645f653d68251ef72ba011"
        }
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice" => {
            "81aca2b6fac304944d8acf345272d8a9a727d5fc2e2e66b222ab4729340c7455"
        }
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base" => {
            "b4f01752d15a488abde3e1ab44723ae4f4b9e68a4037257b098b3737893cc1f9"
        }
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" => {
            "17a07f527a1c25ea30b4e023a184482a23d3e279d697b1dc81b1bde498d29cf9"
        }
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign" => {
            "aecd2cc4c1fe9edef1cb7ca7c401685a43879ad43f3f9e883f1c6760b61731e0"
        }
        _ => panic!("unknown model in config sha map: {model_id}"),
    }
}

fn encode_ids(tokenizer: &Tokenizer, value: &str) -> Vec<u32> {
    tokenizer
        .encode(value, false)
        .expect("tokenizer encode")
        .get_ids()
        .to_vec()
}

fn load_expected_matrix() -> HashMap<String, ModelMatrixEntry> {
    let matrix_path = Path::new("config/model-matrix.json");
    if !matrix_path.exists() {
        panic!(
            "FIXTURE_MISSING: model matrix missing: {}",
            matrix_path.display()
        );
    }
    let matrix_data = std::fs::read_to_string(matrix_path)
        .unwrap_or_else(|err| panic!("FIXTURE_MISSING: read model matrix failed: {err}"));
    let matrix: ModelMatrixConfig =
        serde_json::from_str(&matrix_data).expect("parse model matrix config");
    let mut map = HashMap::new();
    for item in matrix.models {
        map.insert(item.id.clone(), item);
    }
    map
}

fn sha256_file(path: &Path) -> String {
    use std::io::Read;
    let mut file = std::fs::File::open(path).unwrap_or_else(|err| {
        panic!(
            "FIXTURE_MISSING: failed to open file {}: {err}",
            path.display()
        )
    });
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer).expect("read file");
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }

    let result = hasher.finalize();
    format!("{:x}", result)
}

fn load_metadata_from_config(config_path: &Path) -> Result<TestMetadata, String> {
    let config_data =
        std::fs::read_to_string(config_path).map_err(|err| format!("read config failed: {err}"))?;
    let value: Value = serde_json::from_str(&config_data)
        .map_err(|err| format!("parse config json failed: {err}"))?;
    let tts_model_size = value
        .get("tts_model_size")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "missing tts_model_size".to_string())?;
    let tts_model_type = value
        .get("tts_model_type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "missing tts_model_type".to_string())?;

    Ok(TestMetadata {
        tts_model_size: tts_model_size.to_string(),
        tts_model_type: tts_model_type.to_string(),
    })
}

#[derive(Debug)]
struct TestMetadata {
    tts_model_size: String,
    tts_model_type: String,
}

fn model_revision_from_path(model_dir: &Path) -> Option<String> {
    let mut current = Some(model_dir);
    while let Some(dir) = current {
        if let Some(name) = dir.file_name().and_then(|value| value.to_str()) {
            if name.len() == 40 && name.chars().all(|c| c.is_ascii_hexdigit()) {
                return Some(name.to_string());
            }
        }
        current = dir.parent();
    }
    None
}
