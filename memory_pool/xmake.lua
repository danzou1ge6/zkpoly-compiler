add_requires("doctest")
add_rules("mode.debug", "mode.release")

target("memory_pool")
    set_kind("static")
    set_optimize("fastest")
    add_values("cuda.build.devlink", true)
    add_files("cpp/src/*.cu")
    add_files("cpp/wrapper/*.cpp")
    set_languages("c++17")
    set_targetdir("build")

target("test_memory_pool")
    add_packages("doctest")
    if is_mode("release") then
        set_optimize("fastest")
    end
    add_files("cpp/tests/*.cpp")
    add_files("cpp/src/*.cu")
    set_languages("c++17")