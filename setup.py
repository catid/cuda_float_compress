import os
import sys
import subprocess
import multiprocessing
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])

class BuildPackage(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        # FIXME: Release mode fails with an error: undefined symbol: fatbinData
        cfg = 'Debug' if self.debug else 'RelWithDebInfo'
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        script_dir = os.path.dirname(os.path.abspath(__file__))

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]

        build_args = ['--config', cfg]

        num_threads = multiprocessing.cpu_count() - 1
        if num_threads > 1:
            build_args.append(f"-j{num_threads}")

        os.makedirs(self.build_temp, exist_ok=True)
        subprocess.check_call(["cmake", "-S", ".", "-B", self.build_temp] + cmake_args, cwd=script_dir)
        subprocess.check_call(["cmake", "--build", self.build_temp] + build_args, cwd=script_dir)

setup(
    name='cuda_float_compress',
    version='0.2.1',
    python_requires='>=3.7',
    author='catid',
    description='A PyTorch CUDA extension for floating-point compression',

    ext_modules=[CMakeExtension('cuda_float_compress')],
    cmdclass={"build_ext": BuildPackage},
    package_data={
        'cuda_float_compress': ['cuda_float_compress*.so'],
    },
    include_package_data=True,

    zip_safe=False,

    install_requires=[
        'torch',
        'numpy',
    ],
)
