use std::{
    path::{self, Path},
    process::Command,
};
use zkpoly_common::get_project_root::get_project_root;

pub fn resolve_type(name: &str) -> &str {
    match name {
        "halo2curves::bn256::fr::Fr" => "bn254_fr",
        "halo2curves::bn256::fq::Fq" => "bn254_fq",
        "halo2curves::bls12381::fr::Fr" => "bls12381_fr",
        _ => unimplemented!("not implemented for type {}", name),
    }
}

pub fn resolve_curve(name: &str) -> &str {
    match name {
        s if s.starts_with("halo2curves::bn256") => "bn254",
        _ => unimplemented!("unimplemented for curve {}", name),
    }
}

pub fn xmake_run(target: &str) {
    if !Command::new("sh")
        .current_dir(get_project_root())
        .arg("-c")
        .arg(format!("xmake build {}", target))
        .status()
        .expect("could not spawn `xmake`")
        .success()
    {
        // Panic if the command was not successful.
        panic!("could not build the library");
    }
}

pub fn xmake_config(name: &str, value: &str) {
    if !Command::new("sh")
        .current_dir(get_project_root())
        .arg("-c")
        .arg(format!("xmake f --cuda=/usr/local/cuda-10.2 --cu-ccbin=gcc-8 --{}={}", name, value))
        .status()
        .expect("could not spawn `xmake`")
        .success()
    {
        // Panic if the command was not successful.
        panic!("could not set the config");
    }
}

pub fn xmake_run_absolute(target: &str, path: impl AsRef<Path>) {
    if !Command::new("sh")
        .current_dir(path.as_ref())
        .arg("-c")
        .arg(format!("xmake build {}", target))
        .status()
        .expect("could not spawn `xmake`")
        .success()
    {
        // Panic if the command was not successful.
        panic!("could not build the library");
    }
}

pub fn xmake_config_absolute(name: &str, value: &str, path: impl AsRef<Path>) {
    if !Command::new("sh")
        .current_dir(path.as_ref())
        .arg("-c")
        .arg(format!("xmake f --{}={}", name, value))
        .status()
        .expect("could not spawn `xmake`")
        .success()
    {
        // Panic if the command was not successful.
        panic!("could not set the config");
    }
}

pub fn make_run(target: &str, makefile: &str) {
    if !Command::new("make")
        .current_dir(get_project_root())
        .arg("-f")
        .arg(makefile)
        .arg(target)
        .status()
        .expect("could not spawn `make`")
        .success()
    {
        // Panic if the command was not successful.
        panic!("could not build the library");
    }
}
