from setuptools import setup, find_packages

setup(
    name = 'dgd',
    version = '0.0.1',
    packages = find_packages(),
    license='MIT License',
    description = 'Deep genetic designer',
    keywords = [
    'artificial intelligence',
    'biological circuit design'
    ],
    tests_require=[
    'pytest'
    ],
    classifiers=[
    'Development Status :: Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.9',
    ],
)

