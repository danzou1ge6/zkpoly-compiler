use std::env;
use std::path::PathBuf;

fn main() {
    let cuda_path = env::var("CUDA_PATH").unwrap_or("/usr/local/cuda-10.2".to_string());

    let cuda_lib_path = format!("{}/lib64", cuda_path);
    let cuda_header_path = format!("{}/include", cuda_path);
    let cuda_runtime_header_path = format!("{}/include/cuda_runtime.h", cuda_path);
    let cuda_driver_header_path = format!("{}/include/cuda.h", cuda_path);

    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    // Tell cargo to look for libraries in the specified directory
    println!("cargo:rustc-link-search={}", cuda_lib_path);
    println!("cargo:rustc-link-lib=dylib=cudart"); // 使用动态链接
    println!("cargo:rustc-link-lib=dylib=cudadevrt");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header(cuda_runtime_header_path)
        .header(cuda_driver_header_path)
        .clang_arg(format!("-I{}", cuda_header_path))
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
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
