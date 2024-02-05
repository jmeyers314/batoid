import os
import sys
import subprocess

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'RelWithDebInfo'
        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        build_args += ['--', '-j16']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        if 'CMAKE_VERBOSE_MAKEFILE' in env:
            cmake_args += ['-DCMAKE_VERBOSE_MAKEFILE=1']
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


with open("README.rst", 'r') as fh:
    long_description = fh.read()


setup(
    name='batoid',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    author='Josh Meyers',
    author_email='jmeyers314@gmail.com',
    url='https://github.com/jmeyers314/batoid',
    description="Optics raytracer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=['batoid/data', 'batoid/data/*']),
    package_dir={'batoid': 'batoid'},
    package_data={'batoid' : ['data/**/*']},
    ext_modules=[CMakeExtension('batoid._batoid')],
    install_requires=['pybind11', 'numpy', 'pyyaml', 'scipy', 'galsim', 'matplotlib', 'astropy'],
    python_requires='>=3.8',
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    project_urls={
        'Documentation': "https://jmeyers314.github.io/batoid/overview.html",
        'Source': "https://github.com/jmeyers314/batoid",
        'Tracker': "https://github.com/jmeyers314/batoid/issues"
    }
)
