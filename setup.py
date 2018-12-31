from setuptools import setup, find_packages

with open('README.rst') as f:
    long_description = ''.join(f.readlines())

setup(name='gym_windy_gridworlds',
       version='0.0.1',
	  description='Windy Gridworlds environments for OpenAI gym.',
	  long_description=long_description,
	  author='Ibrahim El Shar',
	  author_email='ibrahim.elshar@gmail.com',
	  license='MIT License',
	  url='https://github.com/ibrahim-elshar/gym-windy-gridworlds',
	  packages=find_packages(),
      install_requires=['gym>=0.2.3', 'numpy']
      )
