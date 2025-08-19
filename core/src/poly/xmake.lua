option("POLY_FIELD")
    set_default("bn254_fr") -- 默认值为 TEMPLATE_A
    set_showmenu(true) -- 在 xmake f --help 中显示该选项
    set_description("Choose a field for poly") -- 设置选项的描述信息

target("test_poly_basic")
    add_files("tests/test_basic.cu")
    add_packages("doctest")
    add_cuflags("--extended-lambda")
    add_cugencodes("compute_70")

target("test_poly_eval")
    add_files("tests/test_eval.cu")
    add_cuflags("--extended-lambda")
    add_packages("doctest")
    add_cugencodes("compute_70")

target("test_kate_division")
    add_files("tests/test_kate.cu")
    add_cuflags("--extended-lambda")
    add_packages("doctest")
    add_cugencodes("compute_70")

target("test_invert")
    add_files("tests/test_invert.cu")
    add_cuflags("--extended-lambda")
    add_packages("doctest")
    add_cugencodes("compute_70")

target("poly")
    set_kind("shared")
    set_targetdir(os.projectdir().."/lib")
    add_files("src/*.cu")
    if has_config("POLY_FIELD") then
        add_defines("POLY_FIELD="..get_config("POLY_FIELD").."::Element")
    end
    set_optimize("fastest")
    add_cuflags("--extended-lambda")
    add_cugencodes("compute_70")

target("test_rotate")
    add_files("tests/test_rotate.cu")
    add_cuflags("--extended-lambda")
    add_packages("doctest")
    add_cugencodes("compute_70")

target("test_permute")
    add_files("tests/test_permute.cu")
    add_cuflags("--extended-lambda")
    add_cugencodes("compute_70")
