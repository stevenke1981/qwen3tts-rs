#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionId(pub u128);

pub trait SynthesisSession: Send {
    type Error;
    fn id(&self) -> SessionId;
    fn step(&mut self) -> Result<bool, Self::Error>;
    fn cancel(&self);
    fn is_cancelled(&self) -> bool;
    fn resident_bytes(&self) -> usize;
}

pub trait SessionScheduler {
    type Error;
    fn add(&mut self, session: Box<dyn SynthesisSession<Error = Self::Error>>);
    fn tick(&mut self) -> Result<usize, Self::Error>;
    fn cancel(&mut self, id: SessionId) -> bool;
}
