from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()


setup(name='pyastrotools',
      version='0.2.1',
      description='Set of tools useful for astronomy and optics',
      long_description=readme(),
      url='https://github.com/shbhuk/pyastrotools',
      author='Shubham Kanodia',
      install_requires=[
		  'astropy==4.2',
		  'numpy==1.22.3',
		  'scipy',
		  'astroquery>=0.3.10', 
		  'astroplan',
		  'uncertainties', 
		  'pandas',
		   'matplotlib==3.5.2',
		   # 'WavelengthCalibrationTool @ git+https://github.com/indiajoe/WavelengthCalibrationTool.git@master' # Example from Joe
		   'mrexo @ git+https://github.com/shbhuk/mrexo.git@n_dimensions_generalization', # Need this for the mass prediction
		   # 'TESS_MADNESS @ git+https://github.com/gummiks/TESS_MADNESS.git@master', # Need this for ETC
		   'gdr3bcg @ git+https://gitlab.oca.eu/ordenovic/gaiadr3_bcg.git@main' # Unable to get this to work for now :/
        ],
      author_email='shbhuk@gmail.com',
      license='GPLv3',
      classifiers=['Topic :: Scientific/Engineering :: Astronomy'],
      include_package_data=True
      )

