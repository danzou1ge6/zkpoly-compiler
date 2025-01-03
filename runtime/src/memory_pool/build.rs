use std::env;
use std::path::PathBuf;

fn main() {
    // This is the directory where the `c` library is located.
    let libdir_path = PathBuf::from(".")
        .canonicalize()
        .expect("cannot canonicalize path")
        .join("build/");

    // This is the path to the `c` headers file.
    let headers_path = PathBuf::from("cpp/wrapper/memory_pool_wrapper.h");
    let headers_path_str = headers_path.to_str().expect("Path is not a valid string");

    if !std::process::Command::new("xmake")
        .arg("build")
        .arg("memory_pool")
        .output()
        .expect("could not spawn `xmake`")
        .status
        .success() 
    {
        // Panic if the command was not successful.
        panic!("could not build the library");
    }
    // run twice because the bug of xmake not linking the library sometimes
    if !std::process::Command::new("xmake")
        .arg("build")
        .arg("memory_pool")
        .output()
        .expect("could not spawn `xmake`")
        .status
        .success() 
    {
        // Panic if the command was not successful.
        panic!("could not build the library");
    }

    println!("cargo:rerun-if-changed={}", "cpp/src/*");
    println!("cargo:rerun-if-changed={}", "cpp/wrapper/*");

    // Tell cargo to look for libraries in the specified directory
    println!("cargo:rustc-link-search={}", libdir_path.to_str().unwrap());
    println!("cargo:rustc-link-search=/usr/local/cuda/lib64");

    println!("cargo:rustc-link-lib=static=memory_pool");
    println!("cargo:rustc-link-lib=dylib=cudart"); // 使用动态链接
    println!("cargo:rustc-link-lib=dylib=cudadevrt");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64/stub");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header(headers_path_str)
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .clang_arg("-xc++") // Enable C++ mode
        .clang_arg("-std=c++11") // Specify the C++ standard
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    bindings
        .write_to_file(out_path)
        .expect("Couldn't write bindings!");
}
