target("simple_add")
    set_kind("shared")
    set_targetdir(os.projectdir().."/lib")
    local field = os.getenv("SIMPLE_ADD_FIELD") or "bn254_fr::Element"
    add_files("src/add.cu")
    add_defines("FIELD="..field)