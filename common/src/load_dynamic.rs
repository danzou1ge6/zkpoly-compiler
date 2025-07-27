use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use libloading::Library;

use crate::get_project_root::get_project_root;

#[derive(Debug, Clone)]
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
        let absolute_path = path
            .canonicalize()
            .unwrap_or_else(|_| "".into())
            .to_string_lossy()
            .to_string();

        self.libs.contains_key(&absolute_path)
    }

    pub fn contains_absolute(&self, path: impl AsRef<Path>) -> bool {
        let absolute_path = path
            .as_ref()
            .canonicalize()
            .unwrap_or_else(|_| "".into())
            .to_string_lossy()
            .to_string();

        self.libs.contains_key(&absolute_path)
    }

    pub fn load_relative(&mut self, path: impl AsRef<Path>) -> &'static Library {
        let path = PathBuf::from(get_project_root()).join("lib/").join(path);
        let absolute_path = path.canonicalize().unwrap().to_string_lossy().to_string();

        self.libs
            .entry(absolute_path.clone())
            .or_insert_with(|| Box::leak(Box::new(unsafe { Library::new(path).unwrap() })))
    }

    pub fn load_absolute(&mut self, path: impl AsRef<Path>) -> &'static Library {
        let absolute_path = path
            .as_ref()
            .canonicalize()
            .unwrap()
            .to_string_lossy()
            .to_string();

        self.libs.entry(absolute_path.clone()).or_insert_with(|| {
            Box::leak(Box::new(unsafe {
                Library::new(path.as_ref().as_os_str()).unwrap()
            }))
        })
    }
}
