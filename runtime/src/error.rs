#[derive(Debug)]
pub enum RuntimeError {
    ArgNumWrong,
    TypError,
    VariableTypError,
    Other(String),
}

pub type Result<T> = std::result::Result<T, RuntimeError>;
