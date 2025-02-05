option("FUSED_FIELD")
    set_default("bn254_fr::Element") -- 默认值为 TEMPLATE_A
    set_showmenu(true) -- 在 xmake f --help 中显示该选项
    set_description("Choose a field for ntt") -- 设置选项的描述信息

target("fused_kernels")
    set_kind("shared")
    set_targetdir(os.projectdir().."/lib")
    add_files("src/*.cu")
    if has_config("FUSED_FIELD") then
        add_defines("FUSED_FIELD="..get_config("FUSED_FIELD"))
    end
    add_cugencodes("native")
    set_optimize("fastest")