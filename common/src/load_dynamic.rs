use std::{collections::HashMap, path::PathBuf};

use libloading::Library;

use crate::get_project_root::get_project_root;

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

    pub fn contains(&self, path: &str) -> bool {
        let path = PathBuf::from(get_project_root()).join("lib/").join(path);
        let absolute_path = path.canonicalize().unwrap().to_string_lossy().to_string();

        self.libs.contains_key(&absolute_path)
    }

    pub fn load(&mut self, path: &str) -> &'static Library {
        let path = PathBuf::from(get_project_root()).join("lib/").join(path);
        let absolute_path = path.canonicalize().unwrap().to_string_lossy().to_string();

        self.libs
            .entry(absolute_path.clone())
            .or_insert_with(|| Box::leak(Box::new(unsafe { Library::new(path).unwrap() })))
    }
}
