target("test_bn254")
    if is_mode("debug") then
        set_symbols("debug")
    end
    add_files("./tests/bn254.cu")
    add_cugencodes("native")
    add_packages("doctest")
    set_languages("c++17")