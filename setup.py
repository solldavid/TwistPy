"""
:copyright:
    David Sollberger (david.sollberger@gmail.com), 2022
:license:
    None
"""
from setuptools import setup, find_packages

setup(
    name='twistpy',
    version='0.0',
    description='Toolbox for Wavefield Inertial Sensing',
    url='github.com/solldavid/twistpy',
    author='David Sollberger',
    author_email='david.sollberger@gmail.com',
    license='None',
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
    install_requires=["numpy >= 1.15.0", "scipy >= 1.4.0"],
    test_suite="pytests",
    tests_require=["pytest"]
)
