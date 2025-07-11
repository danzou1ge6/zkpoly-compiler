option("FUSED_FIELD")
    set_default("bn254_fr") -- 默认值为 TEMPLATE_A
    set_showmenu(true) -- 在 xmake f --help 中显示该选项
    set_description("Choose a field for ntt") -- 设置选项的描述信息

target("fused_kernels")
    set_kind("shared")
    set_targetdir(os.projectdir().."/lib")
    add_files("src/*.cu")

    if has_config("FUSED_FIELD") then
        add_defines("FUSED_FIELD="..get_config("FUSED_FIELD").."::Element")
    end

    add_cugencodes("native")
    set_optimize("fastest")
    set_languages("c++17")

    -- 这个钩子在所有源文件准备好后运行，可以修改每个文件的编译选项
    before_build_files(function (target, sourcebatch, opt)
        for _, sourcefile in ipairs(sourcebatch.sourcefiles) do
            -- 提取文件名（不带路径）
            local filename = path.filename(sourcefile)
            -- 匹配结尾数字，如 mykernel_64.cu 中的 64
            local regcount = filename:match("(%d+)%.[^%.]+$")
            if regcount then
                local flag = "--maxrregcount=" .. regcount
                print("Applying to " .. filename .. ": " .. flag)
                -- 为该源文件添加该选项
                -- 256 is the biggest value for maxrregcount, so we don't need to set it
                if tonumber(regcount) < 256 then
                    target:fileconfig_set(sourcefile, {
                        cuflags = flag
                    })
                end
            end
        end
    end)
