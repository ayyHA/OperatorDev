from torch.utils.cpp_extension import BuildExtension,CUDAExtension
from setuptools import setup

setup(
    name = "flashattn",
    ext_modules = [
        CUDAExtension(
            name = "fav1",
            sources = ["flashattn-v1.cu","flashattn.cc"],
            extra_compile_args = {
                "cxx" : ["-g"],
                "nvcc" : ["-O2"]
            }
        )
    ],
    cmdclass = {'build_ext' : BuildExtension}
    # cmdclass = {'build_ext' : BuildExtension.with_options(use_ninja=False)}
    
)