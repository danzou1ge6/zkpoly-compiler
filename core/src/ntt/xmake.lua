
option("NTT_FIELD")
    set_default("bn254_fr") -- 默认值为 TEMPLATE_A
    set_showmenu(true) -- 在 xmake f --help 中显示该选项
    set_description("Choose a field for ntt") -- 设置选项的描述信息

target("ntt")
    set_kind("shared")
    set_targetdir(os.projectdir().."/lib")
    add_files("src/ntt.cu")
    if has_config("NTT_FIELD") then
        add_defines("NTT_FIELD="..get_config("NTT_FIELD").."::Element")
    end
    add_cugencodes("native")
    set_optimize("fastest")
    set_languages("c++17")

target("test_ssip")
    set_kind("binary")
    add_cugencodes("native")
    add_files("tests/test_ssip.cu")
    set_languages("c++17")

target("test_recompute")
    set_kind("binary")
    add_cugencodes("native")
    add_files("tests/test_recompute.cu")
    set_languages("c++17")
