from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

setup_args = dict(
    name='pystoned2',
    version='0.0.1',
    description='A Python Package for Convex Regression and Frontier Estimation',
    long_description_content_type="text/markdown",
    long_description=README,
    license='GPLv3',
    packages=find_packages(),
    author='Shuo Hu',
    author_email='1019753743@qq.com',
    keywords=['StoNED', 'CNLS', 'CER', 'CQR', 'Z-variables','CNLSG'],
    url='https://github.com/advancehs/pyStoNED2',
    # download_url='https://pypi.org/project/pystoned/',
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    package_data={'pystoned': ['data/*.csv']},
)

install_requires = [
    'pyomo>=6.4.4',
    'pandas>=1.1.3',
    'numpy>=1.19.2',
    'scipy>=1.5.2',
    'matplotlib>=3.5.1',
    'mosek>=9.3.18'
]

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)
