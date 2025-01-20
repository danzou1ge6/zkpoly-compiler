
option("NTT_FIELD")
    set_default("bn254_fr::Element") -- 默认值为 TEMPLATE_A
    set_showmenu(true) -- 在 xmake f --help 中显示该选项
    set_description("Choose a field for simple add")

target("ntt")
    set_kind("shared")
    set_targetdir(os.projectdir().."/lib")
    add_files("src/ntt.cu")
    if has_config("NTT_FIELD") then
        add_defines("NTT_FIELD="..get_config("NTT_FIELD"))
    end
    add_cugencodes("native")
    set_optimize("fastest")

target("test_ssip")
    set_kind("binary")
    add_cugencodes("native")
    add_files("tests/test_ssip.cu")

target("test_recompute")
    set_kind("binary")
    add_cugencodes("native")
    add_files("tests/test_recompute.cu")
