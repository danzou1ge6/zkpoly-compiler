"""
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
"""

# 展开上面的targets
import os
from pathlib import Path
import glob

# 配置参数
curve_names = ["bn254", "bls12381"]
curve_scalar_bits = [254, 255]
window_sizes = list(range(8, 32))  # 8到31
alphas = list(range(1, 31))  # 1到30
debugs = [True, False]

# 源文件和目标路径配置
src_file = "core/src/msm/src/msm.cu"
src_dir = os.path.dirname(src_file)
build_dir = "build"
lib_dir = "lib"
cuda_arch = "70"  # 根据你的GPU架构调整
std = "c++14"

# 查找所有相关头文件
def find_header_files(src_dir):
    # 获取源目录及其子目录下的所有头文件
    headers = []
    for ext in ['*.h', '*.hpp', '*.cuh']:
        # headers.extend(glob.glob(f"{src_dir}/**/{ext}", recursive=True))
        headers.extend(glob.glob(f"{src_dir}/{ext}"))
    return headers

# 确保构建目录和库目录存在
os.makedirs(build_dir, exist_ok=True)
os.makedirs(lib_dir, exist_ok=True)
os.makedirs(f"{build_dir}/.objs", exist_ok=True)

# 生成直接的Makefile
def generate_makefile(targets, output_file="Makefile.msm", cuda_path="/usr/local/cuda-10.2", cppc='usr/bin/g++-8'):
    # 查找所有头文件
    header_files = find_header_files(src_dir)
    headers_str = " ".join(header_files)
    
    targets_str = ""
    rules = []
    all_targets = []
    
    for target_config in targets:
        target_name = target_config["name"]
        all_targets.append(f"{target_name}")
        
        # 设置变量
        curve_name = target_config["defines"]["MSM_CURVE"]
        curve_bits = target_config["defines"]["MSM_BITS"]
        window_size = target_config["defines"]["MSM_WINDOW_SIZE"]
        target_windows = target_config["defines"]["MSM_TARGET_WINDOWS"]
        debug = target_config["defines"]["MSM_DEBUG"]
        
        # 创建对象文件目录
        obj_dir = f"{build_dir}/.objs/{target_name}"
        dev_obj_dir = f"{obj_dir}/devlink"
        os.makedirs(obj_dir, exist_ok=True)
        os.makedirs(dev_obj_dir, exist_ok=True)
        
        # 对象文件路径
        obj_file = f"{obj_dir}/msm.cu.o"
        dev_obj_file = f"{dev_obj_dir}/{target_name}_gpucode.cu.o"
        lib_file = f"{lib_dir}/lib{target_name}.so"
        
        # 编译标志
        defines = [
            f"-DMSM_BITS={curve_bits}",
            f"-DMSM_CURVE={curve_name}",
            f"-DMSM_WINDOW_SIZE={window_size}",
            f"-DMSM_TARGET_WINDOWS={target_windows}",
            f"-DMSM_DEBUG={debug}"
        ]
        
        # 包含源目录作为头文件搜索路径
        include_paths = ["-I" + os.path.dirname(src_dir)]
        
        # 步骤1：编译CUDA源文件到目标文件
        compile_cmd = f"$(CUDA_PATH)/bin/nvcc -c -Xcompiler -fPIC -O3 -std={std} -I$(CUDA_PATH)/include {' '.join(include_paths)} {' '.join(defines)} -m64 -rdc=true -gencode arch=compute_{cuda_arch},code=compute_{cuda_arch} -DNDEBUG -o {obj_file} {src_file}"
        
        # 步骤2：设备链接
        devlink_cmd = f"$(CUDA_PATH)/bin/nvcc -o {dev_obj_file} {obj_file} -L$(CUDA_PATH)/lib64 -lcudadevrt -lcudart_static -lrt -lpthread -ldl -m64 -gencode arch=compute_{cuda_arch},code=compute_{cuda_arch} -dlink -shared"
        
        # 步骤3：最终链接
        link_cmd = f"$(CPPC) -o {lib_file} {obj_file} {dev_obj_file} -shared -m64 -fPIC -L$(CUDA_PATH)/lib64 -s -lcudadevrt -lcudart_static -lrt -lpthread -ldl"
        
        # 添加规则，包含头文件依赖
        rules.append(f"{lib_file}: {src_file} {headers_str}")
        rules.append(f"\t@mkdir -p {os.path.dirname(obj_file)}")
        rules.append(f"\t@mkdir -p {os.path.dirname(dev_obj_file)}")
        rules.append(f"\t@echo 'compiling {target_name}'")
        rules.append(f"\t@{compile_cmd}")
        rules.append(f"\t@echo 'devlinking {target_name}'")
        rules.append(f"\t@{devlink_cmd}")
        rules.append(f"\t@echo 'linking {target_name}'")
        rules.append(f"\t@{link_cmd}")
        rules.append(f"\t@echo 'complete {target_name}'")
        rules.append("")
    
    # 写入Makefile
    with open(output_file, "w") as f:
        f.write("# 自动生成的MSM编译Makefile\n\n")
        f.write("# 所有头文件依赖\n")
        f.write(f"HEADERS := {headers_str}\n\n")
        f.write(f"CUDA_PATH = {cuda_path}\n")
        f.write(f"CPPC = {cppc}\n\n")
        f.write(".PHONY: all clean\n\n")
        
        f.write(f"all: {' '.join(all_targets)}\n\n")
        
        for rule in rules:
            f.write(f"{rule}\n")
        
        # 添加各个目标的库文件作为依赖
        lib_targets = [f"{lib_dir}/lib{target}.so" for target in all_targets]
        f.write(f"{' '.join(all_targets)}: {' '.join(lib_targets)}\n\n")
        
        # 清理规则
        f.write("clean:\n")
        f.write(f"\t@rm -rf {build_dir}/.objs\n")
        f.write(f"\t@rm -rf {lib_dir}/libmsm*.so\n")
    
    print(f"generated Makefile: {output_file}")
    print(f"including {len(all_targets)} targets")
    print(f"watching {len(header_files)} header files for changes")
    print("usage:")
    print(f"  make -f {output_file}       # compile all")
    print(f"  make -f {output_file} clean # clear all")

# 生成所有目标的配置
targets = []

# 添加命令行参数支持，允许筛选配置
import argparse
parser = argparse.ArgumentParser(description="Generate MSM compilation Makefile")
parser.add_argument("--curve", choices=curve_names, help="Filter by curve name")
parser.add_argument("--window", type=int, help="Filter by window size")
parser.add_argument("--alpha", type=int, help="Filter by target windows")
parser.add_argument("--debug", action="store_true", help="Generate only debug targets")
parser.add_argument("--no-debug", action="store_true", help="Generate only non-debug targets")
parser.add_argument("--output", default="Makefile.msm", help="Output makefile name")
parser.add_argument("--arch", default=cuda_arch, help="CUDA architecture (default: 80)")
parser.add_argument("--cuda-path", default='/usr/local/cuda-10.2', help="Path to CUDA installation")
parser.add_argument("--cppc", default='/usr/bin/g++-8', help="Path to C++ compiler")

args = parser.parse_args()

# 更新CUDA架构
if args.arch:
    cuda_arch = args.arch

# 根据命令行参数筛选配置
for curve_id, curve_name in enumerate(curve_names):
    if args.curve and args.curve != curve_name:
        continue
    
    for window_size in window_sizes:
        if args.window and args.window != window_size:
            continue
        
        for alpha in alphas:
            if args.alpha and args.alpha != alpha:
                continue
            
            for debug in debugs:
                if args.debug and not debug:
                    continue
                if args.no_debug and debug:
                    continue
                
                target_name = f"msm_{curve_name}_{window_size}_{alpha}_{str(debug).lower()}"
                curve_scalar_bit = curve_scalar_bits[curve_id]
                
                target_config = {
                    "name": target_name,
                    "kind": "shared",
                    "targetdir": "lib",
                    "files": ["src/msm.cu"],
                    "defines": {
                        "MSM_BITS": curve_scalar_bit,
                        "MSM_CURVE": curve_name,
                        "MSM_WINDOW_SIZE": window_size,
                        "MSM_TARGET_WINDOWS": alpha,
                        "MSM_DEBUG": str(debug).lower()
                    },
                    "cugencodes": ["native"],
                    "optimize": "fastest",
                    "languages": ["c++17"]
                }
                
                targets.append(target_config)

# 如果没有找到任何目标，生成一个默认目标
if not targets:
    # 只使用一组参数以便快速测试
    curve_id = 0
    curve_name = curve_names[curve_id]
    window_size = 16
    alpha = 16
    debug = False
    
    target_name = f"msm_{curve_name}_{window_size}_{alpha}_{str(debug).lower()}"
    curve_scalar_bit = curve_scalar_bits[curve_id]
    
    target_config = {
        "name": target_name,
        "kind": "shared",
        "targetdir": "lib",
        "files": ["src/msm.cu"],
        "defines": {
            "MSM_BITS": curve_scalar_bit,
            "MSM_CURVE": curve_name,
            "MSM_WINDOW_SIZE": window_size,
            "MSM_TARGET_WINDOWS": alpha,
            "MSM_DEBUG": str(debug).lower()
        },
        "cugencodes": ["native"],
        "optimize": "fastest",
        "languages": ["c++17"]
    }
    
    targets.append(target_config)
    print("使用默认配置生成Makefile")

# 生成Makefile
generate_makefile(targets, args.output, cuda_path=args.cuda_path, cppc=args.cppc)
