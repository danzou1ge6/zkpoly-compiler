target("test_msm")
    add_files("./tests/test_msm.cu")
    add_cugencodes("native")
-- option("MSM_BITS")
--     set_default("254") -- 默认值为 TEMPLATE_A
--     set_showmenu(true) -- 在 xmake f --help 中显示该选项
--     set_description("Choose a bits for msm") -- 设置选项的描述信息
-- option("MSM_CURVE")
--     set_default("bn254") -- 默认值为 TEMPLATE_A
--     set_showmenu(true) -- 在 xmake f --help 中显示该选项
--     set_description("Choose a curve for msm") -- 设置选项的描述信息

-- option("MSM_WINDOW_SIZE")
--     set_default("16") -- 默认值为 TEMPLATE_A
--     set_showmenu(true) -- 在 xmake f --help 中显示该选项
--     set_description("Choose a window size for msm") -- 设置选项的描述信息

-- option("MSM_TARGET_WINDOWS")
--     set_default("100") -- 默认值为 TEMPLATE_A
--     set_showmenu(true) -- 在 xmake f --help 中显示该选项
--     set_description("Choose precompute target for msm") -- 设置选项的描述信息

-- option("MSM_DEBUG")
--     set_default("0") -- 默认值为 TEMPLATE_A
--     set_showmenu(true) -- 在 xmake f --help 中显示该选项
--     set_description("Choose debug mode for msm") -- 设置选项的描述信息

-- target("msm")
--     set_kind("shared")
--     set_targetdir(os.projectdir().."/lib")
--     add_files("src/msm.cu")
--     if has_config("MSM_BITS") then
--         add_defines("MSM_BITS="..get_config("MSM_BITS"))
--         add_defines("MSM_CURVE="..get_config("MSM_CURVE"))
--         add_defines("MSM_WINDOW_SIZE="..get_config("MSM_WINDOW_SIZE"))
--         add_defines("MSM_TARGET_WINDOWS="..get_config("MSM_TARGET_WINDOWS"))
--         add_defines("MSM_DEBUG="..get_config("MSM_DEBUG"))
--     end
--     add_cugencodes("native")
--     set_optimize("fastest")

local curve_names = {"bn254", "bls12381"}
local curve_scalar_bits = {254, 255}
local window_sizes = {8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}
local alphas = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30}
local debugs = {true, false}
-- all combines of target
for curve_id, curve_names in ipairs(curve_names) do
    for _, window_size in ipairs(window_sizes) do
        for _, alpha in ipairs(alphas) do
            for _, debug in ipairs(debugs) do
                local target_name = "msm_" .. curve_names .. "_" .. window_size .. "_" .. alpha .. "_" .. tostring(debug)
                target(target_name)
                    set_kind("shared")
                    set_targetdir(os.projectdir().."/lib")
                    add_files("src/msm.cu")
                    
                    curve_scalar_bit = curve_scalar_bits[curve_id]
                    add_defines("MSM_BITS="..curve_scalar_bit)

                    add_defines("MSM_CURVE="..curve_names)
                    add_defines("MSM_WINDOW_SIZE="..window_size)
                    add_defines("MSM_TARGET_WINDOWS="..(alpha))
                    add_defines("MSM_DEBUG="..(tostring(debug)))
                    add_cugencodes("native")
                    set_optimize("fastest")
                    set_languages("c++17")
            end
        end
    end
end