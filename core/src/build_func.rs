use std::process::Command;
use zkpoly_common::get_project_root::get_project_root;

pub fn resolve_type(name: &str) -> &str {
    match name {
        "halo2curves::bn256::fr::Fr" => "bn254_fr::Element",
        "halo2curves::bn256::fq::Fq" => "bn254_fq::Element",
        _ => unimplemented!(),
    }
}

pub fn resolve_curve(name: &str) -> (&str, u32) {
    match name {
        s if s.starts_with("halo2curves::bn256") => ("bn254", 254),
        _ => unimplemented!(),
    }
}

pub fn xmake_run(target: &str) {
    if !Command::new("xmake")
        .arg("build")
        .arg(target)
        .status()
        .expect("could not spawn `xmake`")
        .success()
    {
        // Panic if the command was not successful.
        panic!("could not build the library");
    }
    // run twice because the bug of xmake not linking the library sometimes
    if !Command::new("xmake")
        .current_dir(get_project_root())
        .arg("build")
        .arg(target)
        .status()
        .expect("could not spawn `xmake`")
        .success()
    {
        // Panic if the command was not successful.
        panic!("could not build the library");
    }
}

pub fn xmake_config(name: &str, value: &str) {
    if !Command::new("xmake")
        .current_dir(get_project_root())
        .arg("f")
        .arg(format!("--{}={}", name, value))
        .status()
        .expect("could not spawn `xmake`")
        .success()
    {
        // Panic if the command was not successful.
        panic!("could not set the config");
    }
}
