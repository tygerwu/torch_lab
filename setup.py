import torch
import subprocess 
from setuptools import setup,Extension,find_packages 
import os
import sys 

CWD=os.path.dirname(os.path.abspath(__file__))
BUILD_DIR=os.path.join(CWD,"build")
TORCH_LAB_LIB='libtorch_lab.so'
TORCH_LAB_LIB_PATH=''
PACKAGE='torch_lab'
PACKAGE_DIR=os.path.join(CWD,PACKAGE)
CUTLASS_ROOT=None
CUDA_ARCH='sm86'


def parse_args():
    global BUILD_DIR,TORCH_LAB_LIB_PATH,CUTLASS_ROOT,CUDA_ARCH
    
    # extract the user-defined arguments after the sutup args
    def _retreive_arg(arg_name):
        arg_value = None 
        if arg_name in sys.argv:
            index = sys.argv.index(arg_name)
            if index+1 < len(sys.argv):
                arg_value = sys.argv[index+1]
            
            # remove the arg from the sys.argv so it won't be parsed by setup()
            sys.argv = sys.argv[:index] + sys.argv[index+2:]
        return arg_value

    build_dir = _retreive_arg("--build-dir")
    cutlass_root = _retreive_arg("--cutlass-root")
    cuda_arch = _retreive_arg("--cuda-arch")
    
    BUILD_DIR = build_dir if build_dir is not None else BUILD_DIR
    TORCH_LAB_LIB_PATH = os.path.join(BUILD_DIR,TORCH_LAB_LIB)
    CUTLASS_ROOT = cutlass_root if cutlass_root is not None else None
    CUDA_ARCH = cuda_arch if cuda_arch is not None else CUDA_ARCH
    
    assert CUTLASS_ROOT is not None 
    
def _run_cmd(cmd_str,msg):
    print(msg,cmd_str)
    res = subprocess.run(cmd_str,shell=True)
    assert res.returncode == 0
    
def build():
    TORCH_LIB_DIR = os.path.dirname(torch.__file__)
    TORCH_CMAKE_DIR = os.path.join(TORCH_LIB_DIR,'share','cmake','Torch')
    CMAKE_FILE_DIR = os.path.join(CWD,'cxx')
    
    if not os.path.exists(BUILD_DIR):
        os.makedirs(BUILD_DIR)
        
    
    # prepare cmake
    build_cmd = 'cd {build_dir} && cmake -B {build_dir} {cmake_file} '\
                 '-DCUTLASS_ROOT={cutlass_root} -DTorch_DIR={torch_dir} -DCUDA_ARCH={arch}' \
                 .format(build_dir=BUILD_DIR,cmake_file=CMAKE_FILE_DIR,cutlass_root=CUTLASS_ROOT,\
                         torch_dir=TORCH_CMAKE_DIR,arch=CUDA_ARCH)
    _run_cmd(build_cmd,"build_cmd:")
    
    
    # build so 
    buildso_cmd = 'cd {build_dir} && cmake --build . -j 12 -v' \
                  .format(build_dir=BUILD_DIR) 
    
    _run_cmd(buildso_cmd,"buildso_cmd:")
    
    # cp so into package
    cp_cmd = 'cp {src} {dst}'\
            .format(src=TORCH_LAB_LIB_PATH,dst=PACKAGE_DIR)
            
    _run_cmd(cp_cmd,"cp_cmd:")

def clear():
    package_lib = os.path.join(PACKAGE_DIR,TORCH_LAB_LIB)
    if os.path.exists(package_lib):
        os.remove(package_lib) 

parse_args()
build()

setup(
    name='torch_lab',
    packages = find_packages(where='.'),
    package_data={
        'torch_lab': [TORCH_LAB_LIB]
    }
)

clear()