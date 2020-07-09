from setuptools import setup, find_packages

setup(name='pulseanalysis',
	version='0.1',
	description='Code for analyzing pulse shapes from MKID data',
	url='https://github.com/millerja14/pulseanalysis',
	author='Jacob Miller',
	license='GNU GPLv3',
	packages=find_packages(),
	intall_requires=["numpy", "scipy", "matplotlib", "mkidcalculator"],
	zip_safe=False)
