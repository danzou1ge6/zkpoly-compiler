pub fn resolve_type(name: &str) -> String {
    match name {
        "halo2curves::bn256::fr::Fr" => "bn254_fr:\\:Element".to_string(), // the \\ is because xmake will turn :: into :, so we should add \ to disable it
        "halo2curves::bn256::fq::Fq" => "bn254_fq:\\:Element".to_string(),
        _ => unimplemented!(),
    }
}
