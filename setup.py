from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()


setup(name='pyastrotools',
      version='0.2',
      description='Set of tools useful for astronomy and optics',
      long_description=readme(),
      url='https://github.com/shbhuk/pyastrotools',
      author='Shubham Kanodia',
      install_requires=['astropy>=4.0.4','numpy>=1.17.2','scipy','astroquery>=0.3.10', 'astroplan','uncertainties', 'pandas'],
      author_email='shbhuk@gmail.com',
      license='GPLv3',
      classifiers=['Topic :: Scientific/Engineering :: Astronomy'],      
      include_package_data=True
      )
