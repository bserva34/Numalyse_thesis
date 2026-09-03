# setup.py
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import cv2

# Récupération des flags de compilation OpenCV
import subprocess
opencv_compile_args = subprocess.check_output(['pkg-config', '--cflags', 'opencv4']).decode('utf-8').split()
opencv_link_args = subprocess.check_output(['pkg-config', '--libs', 'opencv4']).decode('utf-8').split()

ext_modules = [
    Pybind11Extension(
        "video_processor_cxx",
        ["process.cpp"],
        extra_compile_args=opencv_compile_args + ["-O3"], # Optimisation max
        extra_link_args=opencv_link_args,
    ),
]

setup(
    name="video_processor_cxx",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)