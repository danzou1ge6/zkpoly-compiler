#[derive(Debug)]
pub enum RuntimeError {
    ArgNumWrong,
    TypError,
    Other(String),
}
