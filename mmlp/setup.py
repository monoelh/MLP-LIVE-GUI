from setuptools import setup

setup(name='MMLP',
      description='numpy only neural network framework',
      long_description='numpy only neural network framework',
      version='0.31',
      url='https://github.com/monoelh/MMLP',
      author='Manuel Hass',
      author_email='manuel.hass@outlook.com',
      packages=['mmlp'],
      zip_safe=False,
      install_requires=[
          'numpy'
      ]
)