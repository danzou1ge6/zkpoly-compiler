add_requires("doctest")
add_rules("mode.debug", "mode.release")

target("test-mont")
    set_languages(("c++17"))
    if is_mode("debug") then
        set_symbols("debug")
    end
    add_files("tests/main.cu")
    add_packages("doctest")

target("bench-mont")
    set_languages(("c++17"))
    add_cugencodes("native")
    add_options("-lineinfo")
    add_options("--expt-relaxed-constexpr")
    add_files("tests/bench.cu")