target("test_msm")
    add_files("./tests/test_msm.cu")
    add_cugencodes("native")
option("MSM_BITS")
    set_default("254") -- 默认值为 TEMPLATE_A
    set_showmenu(true) -- 在 xmake f --help 中显示该选项
    set_description("Choose a bits for msm") -- 设置选项的描述信息
option("MSM_CURVE")
    set_default("bn254") -- 默认值为 TEMPLATE_A
    set_showmenu(true) -- 在 xmake f --help 中显示该选项
    set_description("Choose a curve for msm") -- 设置选项的描述信息

option("MSM_WINDOW_SIZE")
    set_default("16") -- 默认值为 TEMPLATE_A
    set_showmenu(true) -- 在 xmake f --help 中显示该选项
    set_description("Choose a window size for msm") -- 设置选项的描述信息

option("MSM_TARGET_WINDOWS")
    set_default("100") -- 默认值为 TEMPLATE_A
    set_showmenu(true) -- 在 xmake f --help 中显示该选项
    set_description("Choose precompute target for msm") -- 设置选项的描述信息

option("MSM_DEBUG")
    set_default("0") -- 默认值为 TEMPLATE_A
    set_showmenu(true) -- 在 xmake f --help 中显示该选项
    set_description("Choose debug mode for msm") -- 设置选项的描述信息

target("msm")
    set_kind("shared")
    set_targetdir(os.projectdir().."/lib")
    add_files("src/msm.cu")
    if has_config("MSM_BITS") then
        add_defines("MSM_BITS="..get_config("MSM_BITS"))
        add_defines("MSM_CURVE="..get_config("MSM_CURVE"))
        add_defines("MSM_WINDOW_SIZE="..get_config("MSM_WINDOW_SIZE"))
        add_defines("MSM_TARGET_WINDOWS="..get_config("MSM_TARGET_WINDOWS"))
        add_defines("MSM_DEBUG="..get_config("MSM_DEBUG"))
    end
    add_cugencodes("native")
    set_optimize("fastest")