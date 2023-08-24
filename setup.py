import os
import pkg_resources
from setuptools import setup, find_packages


setup(
    name="CRAFT",
    py_modules=["CRAFT"],
    version="1.0",
    description="",
    author="Igor Pavlov, Youngmin Baek, Bado Lee, Dongyoon Han, Sangdoo Yun, Hwalsuk Lee",
    url='https://github.com/boomb0om/CRAFT-text-detection',
    packages=find_packages(include=['CRAFT', 'CRAFT.*']),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ]
)