use std::path::PathBuf;
use std::env;

pub fn get_project_root() -> String {
    let path = env::var("ZKPOLY_COMPILER_PROJECT_ROOT").unwrap_or_else(|_| {
        panic!("ZKPOLY_COMPILER_PROJECT_ROOT is not set");
    });
    let absolute_path = PathBuf::from(path)
            .canonicalize()
            .unwrap()
            .to_string_lossy()
            .to_string();
    absolute_path
}