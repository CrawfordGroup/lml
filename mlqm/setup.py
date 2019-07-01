import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name='mlqm',
        version="0.1.1",
        description='Machine-Learning Quantum Mechanics Software Package',
        author='Benjamin G. Peyton',
        author_email='bgpeyton@vt.edu',
        url="https://github.com/bgpeyton/lml/",
        license='BSD-3C',
        packages=setuptools.find_packages(),
        install_requires=[
            'numpy>=1.7',
        ],
        extras_require={
            'docs': [
                'sphinx==1.2.3',  # autodoc was broken in 1.3.1
                'sphinxcontrib-napoleon',
                'sphinx_rtd_theme',
                'numpydoc',
            ],
            'tests': [
                'pytest',
                'pytest-cov',
            ],
            'datagen': [
                'psi4>=1.3',
            ]
        },

        tests_require=[
            'pytest',
            'pytest-cov',
        ],

        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3',
        ],
        zip_safe=True,
    )
