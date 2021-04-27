from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()


setup(name='pyastrotools',
      version='0.1',
      description='Set of tools useful for astronomy and optics',
      long_description=readme(),
      url='https://github.com/shbhuk/pyastrotools',
      author='Shubham Kanodia',
      author_email='shbhuk@gmail.com',
      # install_requires=['astropy>=4.0.4','jplephem','numpy>=1.17.2','scipy','astroquery>=0.3.10'],
      # packages=['barycorrpy', 'barycorrpy.tests'],
      # license='GPLv3',
      # classifiers=['Topic :: Scientific/Engineering :: Astronomy'],
      # keywords='Barycentric Correction Astronomy Spectroscopy Radial Velocity',
      include_package_data=True
      )
