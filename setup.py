from distutils.core import setup, Extension
import numpy.distutils.misc_util
import numpy as np
import os

command = 'git log -1 | grep ^commit | cut -d " " -f 2 > version_hash'

x = os.system(command)

setup(	name='UV_sim', 
	version="0.1",
	author='Tom Louden',
	author_email = 't.louden@warwick.ac.uk',
	url = 'https://github.com/tomlouden/UV_sim',
	packages =['UV_sim'],
	license = ['GNU GPLv3'],
	description ='uv sim',
	classifiers = [
		'Development Status :: 4 - Beta',
		'Intended Audience :: Science/Research',
		'Topic :: Scientific/Engineering',
		'Programming Language :: Python'
		],
	include_dirs = [np.get_include()],
	install_requires = ['numpy'],
)
