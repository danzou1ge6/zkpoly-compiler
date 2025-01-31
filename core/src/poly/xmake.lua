option("POLY_FIELD")
    set_default("bn254_fr::Element") -- 默认值为 TEMPLATE_A
    set_showmenu(true) -- 在 xmake f --help 中显示该选项
    set_description("Choose a field for poly") -- 设置选项的描述信息

target("test_poly_basic")
    add_files("tests/test_basic.cu")
    add_cugencodes("native")
    add_packages("doctest")

target("test_poly_eval")
    add_files("tests/test_eval.cu")
    add_cugencodes("native")
    add_cuflags("--extended-lambda")

target("test_kate_division")
    add_files("tests/test_kate.cu")
    add_cugencodes("native")
    add_cuflags("--extended-lambda")
    add_packages("doctest")

target("test_scan")
    add_files("tests/test_scan.cu")
    add_cugencodes("native")
    add_cuflags("--extended-lambda")
    add_packages("doctest")

target("test_invert")
    add_files("tests/test_invert.cu")
    add_cugencodes("native")
    add_cuflags("--extended-lambda")
    add_packages("doctest")

target("poly")
    set_kind("shared")
    set_targetdir(os.projectdir().."/lib")
    add_files("src/poly.cu")
    if has_config("POLY_FIELD") then
        add_defines("POLY_FIELD="..get_config("POLY_FIELD"))
    end
    add_cugencodes("native")
    set_optimize("fastest")
    add_cuflags("--extended-lambda")

target("test_rotate")
    add_files("tests/test_rotate.cu")
    add_cugencodes("native")
    add_cuflags("--extended-lambda")
    add_packages("doctest")