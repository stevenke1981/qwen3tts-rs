//! Stage-dump observer and file-based manifest writer for parity instrumentation.
//!
//! The production API can keep a no-op observer in default builds, while a writer
//! is passed through explicitly only when `--features stage-dump` is enabled.

#[cfg(feature = "stage-dump")]
use std::collections::HashSet;
#[cfg(feature = "stage-dump")]
use std::io::{self, Write};
#[cfg(feature = "stage-dump")]
use std::path::{Path, PathBuf};

#[cfg(feature = "stage-dump")]
use candle_core::{DType, Device, Error as CandleError};
use candle_core::{Result as CandleResult, Tensor};
use serde::{Deserialize, Serialize};
#[cfg(feature = "stage-dump")]
use sha2::{Digest, Sha256};

#[cfg(feature = "stage-dump")]
use crate::{Error, Result};

/// Metadata required by a stage-dump manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageDumpMetadata {
    pub source: String,
    pub revision: Option<String>,
    pub model: String,
    pub case_id: String,
    pub seed: Option<i64>,
}

/// Top-level stage-dump manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageDumpManifest {
    pub schema_version: u32,
    pub source: String,
    pub revision: Option<String>,
    pub model: String,
    pub case_id: String,
    pub seed: Option<i64>,
    pub stages: Vec<StageDumpEntry>,
}

/// One dumped tensor stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageDumpEntry {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub file: String,
    pub sha256: String,
    pub layout: String,
}

/// Stage hook contract.
///
/// Default implementations are no-ops so non-dump builds only pay minimal call
/// overhead.
pub trait StageDumpObserver {
    fn wants_capture(&self) -> bool {
        false
    }

    fn on_talker_codebook0_logits(
        &mut self,
        _frame_index: usize,
        _logits: &Tensor,
    ) -> CandleResult<()> {
        Ok(())
    }

    fn on_talker_final_codes(&mut self, _codes: &Tensor) -> CandleResult<()> {
        Ok(())
    }

    fn on_code_predictor_step_logits(
        &mut self,
        _frame_index: usize,
        _step: usize,
        _logits: &Tensor,
    ) -> CandleResult<()> {
        Ok(())
    }

    fn on_code_predictor_final_codes(
        &mut self,
        _frame_index: usize,
        _codes: &Tensor,
    ) -> CandleResult<()> {
        Ok(())
    }

    fn on_codec_input_codes(&mut self, _input_codes: &Tensor) -> CandleResult<()> {
        Ok(())
    }

    fn on_codec_output_pcm(&mut self, _pcm: &Tensor) -> CandleResult<()> {
        Ok(())
    }

    fn commit(&mut self) -> CandleResult<()> {
        Ok(())
    }
}

/// Default no-op observer for default builds.
#[derive(Debug, Default)]
pub struct NoopStageDumpObserver;

impl StageDumpObserver for NoopStageDumpObserver {}

/// Session-owned stage dump writer.
#[cfg(feature = "stage-dump")]
#[derive(Debug)]
pub struct StageDumpWriter {
    output_dir: PathBuf,
    manifest_path: PathBuf,
    manifest: StageDumpManifest,
    stage_names: HashSet<String>,
    stage_count: usize,
}

#[cfg(feature = "stage-dump")]
impl StageDumpWriter {
    fn io_error(stage: &str, path: &Path, err: io::Error) -> Error {
        Error::Config(format!(
            "stage-dump {stage} write failed at {}: {err}",
            path.display()
        ))
    }

    pub fn new<P: AsRef<Path>>(output_dir: P, metadata: StageDumpMetadata) -> Result<Self> {
        let output_dir = output_dir.as_ref().to_path_buf();

        if output_dir.exists() {
            return Err(Error::Config(format!(
                "stage-dump output directory already exists: {}",
                output_dir.display()
            )));
        }

        std::fs::create_dir_all(&output_dir).map_err(Error::Io)?;
        let manifest_path = output_dir.join("manifest.json");
        if manifest_path.exists() {
            return Err(Error::Config(format!(
                "stage-dump manifest already exists: {}",
                manifest_path.display()
            )));
        }

        Ok(Self {
            output_dir,
            manifest_path,
            manifest: StageDumpManifest {
                schema_version: 1,
                source: metadata.source,
                revision: metadata.revision,
                model: metadata.model,
                case_id: metadata.case_id,
                seed: metadata.seed,
                stages: Vec::new(),
            },
            stage_names: HashSet::new(),
            stage_count: 0,
        })
    }

    pub fn manifest_path(&self) -> &Path {
        &self.manifest_path
    }

    pub fn manifest(&self) -> &StageDumpManifest {
        &self.manifest
    }

    fn sanitize_stage_name(name: &str) -> Result<String> {
        if name.is_empty() {
            return Err(Error::Config("stage name must not be empty".into()));
        }
        if name.contains("..") || name.contains('/') || name.contains('\\') {
            return Err(Error::Config(format!(
                "unsafe stage name: {name} (contains separator or parent reference)"
            )));
        }
        if !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            return Err(Error::Config(format!(
                "unsafe stage name: {name} (contains unsupported characters)"
            )));
        }
        Ok(name.to_string())
    }

    fn next_stage_path(&mut self, name: &str) -> PathBuf {
        let path = self
            .output_dir
            .join(format!("{name}_{:04}.f32.bin", self.stage_count));
        self.stage_count += 1;
        path
    }

    /// Record a stage tensor using the canonical F32 layout.
    pub fn record_stage(&mut self, name: &str, tensor: &Tensor, layout: &str) -> Result<()> {
        let name = Self::sanitize_stage_name(name)?;
        if self.stage_names.contains(&name) {
            return Err(Error::Config(format!("duplicate stage name: {name}")));
        }

        let stage_path = self.next_stage_path(&name);
        if stage_path.exists() {
            return Err(Error::Config(format!(
                "stage file already exists: {}",
                stage_path.display()
            )));
        }

        let tensor = tensor.to_dtype(DType::F32)?;
        let tensor = tensor.to_device(&Device::Cpu)?;
        let tensor = tensor.contiguous()?;
        let shape = tensor.shape().dims().to_vec();
        let flat = tensor.flatten_all()?.to_vec1::<f32>()?;

        let mut hasher = Sha256::new();
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&stage_path)
            .map_err(|err| Self::io_error(&name, &stage_path, err))?;

        for value in &flat {
            let bytes = value.to_le_bytes();
            hasher.update(bytes);
            file.write_all(&bytes)
                .map_err(|err| Self::io_error(&name, &stage_path, err))?;
        }

        let file_name = stage_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| Error::Config("stage file name is not valid utf-8".into()))?
            .to_string();

        self.stage_names.insert(name.clone());
        self.manifest.stages.push(StageDumpEntry {
            name,
            dtype: "f32".into(),
            shape,
            file: file_name,
            sha256: format!("{:x}", hasher.finalize()),
            layout: layout.to_string(),
        });

        Ok(())
    }

    pub fn write_manifest(&mut self) -> Result<()> {
        if self.manifest_path.exists() {
            return Err(Error::Config(format!(
                "manifest already exists: {}",
                self.manifest_path.display()
            )));
        }

        let payload = serde_json::to_string_pretty(&self.manifest).map_err(Error::Serde)?;
        let mut output = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&self.manifest_path)
            .map_err(|err| Self::io_error("manifest", &self.manifest_path, err))?;
        output
            .write_all(payload.as_bytes())
            .map_err(|err| Self::io_error("manifest", &self.manifest_path, err))?;

        Ok(())
    }
}

#[cfg(feature = "stage-dump")]
impl StageDumpObserver for StageDumpWriter {
    fn wants_capture(&self) -> bool {
        true
    }

    fn on_talker_codebook0_logits(
        &mut self,
        frame_index: usize,
        logits: &Tensor,
    ) -> CandleResult<()> {
        self.record_stage(
            &format!("talker_codebook0_logits_{frame_index:04}"),
            logits,
            "C",
        )
        .map_err(|err| CandleError::Msg(err.to_string()))
    }

    fn on_talker_final_codes(&mut self, codes: &Tensor) -> CandleResult<()> {
        self.record_stage("talker_final_code_matrix", codes, "C")
            .map_err(|err| CandleError::Msg(err.to_string()))
    }

    fn on_code_predictor_step_logits(
        &mut self,
        frame_index: usize,
        step: usize,
        logits: &Tensor,
    ) -> CandleResult<()> {
        self.record_stage(
            &format!("code_predictor_step_logits_{frame_index:04}_{step:04}"),
            logits,
            "C",
        )
        .map_err(|err| CandleError::Msg(err.to_string()))
    }

    fn on_code_predictor_final_codes(
        &mut self,
        frame_index: usize,
        codes: &Tensor,
    ) -> CandleResult<()> {
        self.record_stage(
            &format!("code_predictor_final_code_matrix_{frame_index:04}"),
            codes,
            "C",
        )
        .map_err(|err| CandleError::Msg(err.to_string()))
    }

    fn on_codec_input_codes(&mut self, input_codes: &Tensor) -> CandleResult<()> {
        self.record_stage("codec_input_code_matrix", input_codes, "C")
            .map_err(|err| CandleError::Msg(err.to_string()))
    }

    fn on_codec_output_pcm(&mut self, pcm: &Tensor) -> CandleResult<()> {
        self.record_stage("codec_final_pcm", pcm, "C")
            .map_err(|err| CandleError::Msg(err.to_string()))
    }

    fn commit(&mut self) -> CandleResult<()> {
        self.write_manifest()
            .map_err(|err| CandleError::Msg(format!("failed to write manifest: {err}")))
    }
}

#[cfg(all(test, feature = "stage-dump"))]
mod tests {
    use super::*;

    #[cfg(feature = "stage-dump")]
    #[test]
    fn sanitize_stage_name_rejects_unsafe_inputs() {
        assert!(StageDumpWriter::sanitize_stage_name("").is_err());
        assert!(StageDumpWriter::sanitize_stage_name("..").is_err());
        assert!(StageDumpWriter::sanitize_stage_name("../foo").is_err());
        assert!(StageDumpWriter::sanitize_stage_name("foo/bar").is_err());
        assert!(StageDumpWriter::sanitize_stage_name("foo\\bar").is_err());
        assert!(StageDumpWriter::sanitize_stage_name("foo..bar").is_err());
        assert!(StageDumpWriter::sanitize_stage_name("abc-123_X").is_ok());
    }
}
