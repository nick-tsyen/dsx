from setuptools import setup
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()


setup(name='dsx',
      version='0.9.4.0',
      description='The utilities pack for data science and analytics task. '

                  'The core module ds_utils (Data Science Utilities) is designed to work with Pandas to simplify common tasks, '

                  'such as generating metadata for the dataframe, validating merged dataframe, and visualizing dataframe. ',

      url='https://dataninja.ml/data-analytics/dsx',
      download_url = 'https://github.com/NicTsyen/dsx',

      author='NicTsyen',
      author_email='support@nicdatalab.com',
      license='GNU GENERAL PUBLIC LICENSE',

      install_requires=['joblib', 'seaborn', 'pandas', 'numpy', 'scipy', 'matplotlib', 'regex'],

      py_modules=['dsx.ds_utils', 'dsx.ml_utils'],


      classifiers=['Development Status :: 4 - Beta',
                   'Intended Audience :: Developers',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.6',
                   'Programming Language :: Python :: 3.7'],

      long_description=README,
      long_description_content_type="text/markdown",
      )