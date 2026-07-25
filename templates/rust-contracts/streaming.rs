// Design contract only. Adapt into the production module; do not copy blindly.

pub type CodecFrame = [u16; 16];

#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub frame_index: u64,
    pub sample_rate: u32,
    pub samples: Vec<f32>,
    pub is_final: bool,
}

pub trait CodecStream: Send {
    type Error;

    /// Decode exactly one new frame without re-running prior frames.
    fn decode_frame(&mut self, frame: &CodecFrame) -> Result<AudioChunk, Self::Error>;

    /// Prime with ICL reference codes through the same stateful path.
    fn prime<I>(&mut self, frames: I) -> Result<(), Self::Error>
    where
        I: IntoIterator<Item = CodecFrame>;

    /// Zero all persistent state. A replay after reset must match a new session.
    fn reset(&mut self) -> Result<(), Self::Error>;

    fn frame_position(&self) -> u64;
    fn resident_state_bytes(&self) -> usize;
}
