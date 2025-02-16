#[derive(Debug)]
pub enum RuntimeError {
    ArgNumWrong,
    TypError,
    Other(String),
}

pub type Result<T> = std::result::Result<T, RuntimeError>;
