from setuptools import setup

setup(name='MLP3',
      description='numpy only neural network 3 layer framework',
      long_description='numpy only neural network 3 layer framework',
      version='0.31',
      url='https://github.com/monoelh/MML',
      author='Manuel Hass',
      author_email='manuel.hass@outlook.com',
      packages=['mlp3'],
      zip_safe=False,
      install_requires=[
          'numpy'
      ]
)