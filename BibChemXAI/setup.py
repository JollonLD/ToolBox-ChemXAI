from setuptools import setup, find_packages

setup(
    name='ChemXAI',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'lime==0.2.0.1',
        'rdkit==2024.9.6',
        'scikit-learn==1.6.1',
        'scipy==1.15.2',
        'shap==0.47.0',
        'torch==2.6.0',
        'torch-geometric==2.6.1',
        'ase==3.24.0',
        'dscribe==2.1.1',
    ],
    author='Jonas Lucas Durão',
    author_email='jonas.durao@unifesp.br',
    description='ToolBox made to explain models using Chemistry Datasets',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/JollonLD/ToolBox-ChemXAI.git',
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)