use std::process::Command;

pub fn resolve_type(name: &str) -> &str {
    match name {
        "halo2curves::bn256::fr::Fr" => "bn254_fr::Element",
        "halo2curves::bn256::fq::Fq" => "bn254_fq::Element",
        _ => unimplemented!(),
    }
}

pub fn xmake_run(target: &str) {
    if !Command::new("xmake")
        .arg("build")
        .arg(target)
        .output()
        .expect("could not spawn `xmake`")
        .status
        .success()
    {
        // Panic if the command was not successful.
        panic!("could not build the library");
    }
    // run twice because the bug of xmake not linking the library sometimes
    if !Command::new("xmake")
        .arg("build")
        .arg(target)
        .output()
        .expect("could not spawn `xmake`")
        .status
        .success()
    {
        // Panic if the command was not successful.
        panic!("could not build the library");
    }
}

pub fn xmake_config(name: &str, value: &str) {
    if !Command::new("xmake")
        .arg("f")
        .arg(format!("--{}={}", name, value))
        .output()
        .expect("could not spawn `xmake`")
        .status
        .success()
    {
        // Panic if the command was not successful.
        panic!("could not set the config");
    }
}
