use std::process::Command;

pub fn run_xmake(target: &str) {
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
