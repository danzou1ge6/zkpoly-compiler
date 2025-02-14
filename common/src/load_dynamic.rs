use std::{collections::HashMap, path::PathBuf};

use libloading::Library;

#[derive(Debug)]
pub struct Libs {
    libs: HashMap<String, &'static Library>,
}

impl Libs {
    pub fn new() -> Self {
        Self {
            libs: HashMap::new(),
        }
    }

    pub fn load(&mut self, path: &str) -> &'static Library {
        let absolute_path = PathBuf::from(path)
            .canonicalize()
            .unwrap()
            .to_string_lossy()
            .to_string();

        self.libs
            .entry(absolute_path.clone())
            .or_insert_with(|| Box::leak(Box::new(unsafe { Library::new(path).unwrap() })))
    }
}
