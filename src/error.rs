use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum AegisError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("invalid execution plan: {0}")]
    InvalidPlan(String),
    #[error("unsupported operation: {0}")]
    Unsupported(String),
    #[error("missing required file: {0}")]
    MissingFile(PathBuf),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, AegisError>;
