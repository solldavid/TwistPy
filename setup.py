"""
:copyright:
    David Sollberger (david.sollberger@gmail.com), 2022
:license:
    None
"""
from setuptools import setup, find_packages

setup(
    name='twistpy',
    version='0.0.1',
    description='Toolbox for Wavefield Inertial Sensing',
    url='github.com/solldavid/twistpy',
    author='The TwistPy Developers',
    maintainer='David Sollberger',
    maintainer_email='david.sollberger@gmail.com',
    license='LGPLv3',
    packages=find_packages(),
    platforms='OS Independent',
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Science / Research',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Natural Language :: English'],
    install_requires=["numpy", "scipy"],
    test_suite="pytests",
    tests_require=["pytest"]
)
