import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

data_files_to_include = [('', ['README.md', 'LICENSE'])]

setuptools.setup(
    name='thym_mat_project',
    url="https://github.com/camacesco/thym-mat-project",
    author="Francesco Camaglia",
    author_email="francesco.camaglia@phys.ens.fr",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version='0.0.3',
    description='Python package to standard statbiophys analysis.',
    license="GNU GPLv3",
    python_requires='>=3.5',
    install_requires = [
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "mpmath",
        "kamapack",
        #"sonnia",
    ],
    packages=setuptools.find_packages(),
    data_files = data_files_to_include,
    include_package_data=True,
    entry_points= {
        'console_scripts' : [
            'thym-mat-ngram_entropy=thymmatu.analysis.ngram_entropy:main',
            'thym-mat-folder_stat=thymmatu.analysis.basket_blind_stat:main'
        ],
    }
)
