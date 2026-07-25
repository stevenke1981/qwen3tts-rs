pub type CodecFrame = [u16; 16];

#[derive(Debug)]
pub enum GenerationEvent {
    PromptReady { token_count: usize },
    CodecFrame { index: u64, codes: CodecFrame },
    Audio { index: u64, samples: Vec<f32>, sample_rate: u32 },
    Finished { frames: u64, samples: u64 },
}

pub trait EventSink: Send {
    type Error;
    fn on_event(&mut self, event: GenerationEvent) -> Result<(), Self::Error>;
    fn is_cancelled(&self) -> bool;
}
