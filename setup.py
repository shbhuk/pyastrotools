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
      include_package_data=True
      )
