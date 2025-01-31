-- 定义一个自定义选项
option("SIMPLE_ADD_FIELD")
    set_default("bn254_fr::Element") -- 默认值为 TEMPLATE_A
    set_showmenu(true) -- 在 xmake f --help 中显示该选项
    set_description("Choose a field for simple add")

target("simple_add")
    set_kind("shared")
    set_targetdir(os.projectdir().."/lib")
    set_optimize("fastest")

    if has_config("SIMPLE_ADD_FIELD") then
        add_defines("FIELD="..get_config("SIMPLE_ADD_FIELD"))
    end

    add_files("src/add.cu")