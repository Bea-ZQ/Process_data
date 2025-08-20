from setuptools import setup, find_packages

VERSION = '1.1'
DESCRIPTION = 'Process data package'
LONG_DESCRIPTION = 'Process data package'


# cdflib 1.3.4
# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="Process_data",
        version=VERSION,
        author="BZQ",
        author_email="<beatriz.zenteno@ug.uchile.cl>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that
        # needs to be installed along with your package. Eg: 'caer'
)
