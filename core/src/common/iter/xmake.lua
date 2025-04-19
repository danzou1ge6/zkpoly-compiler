target("test_slice_iter")
    if is_mode("debug") then
        set_symbols("debug")
    end
    add_files("./tests/test_slice_iter.cu")
    add_cugencodes("native")
    add_packages("doctest")
    set_languages("c++17")