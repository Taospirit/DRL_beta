from setuptools import setup, find_packages

setup(
    name = 'drl',
    version = '0.0.1',
    packages = find_packages(exclude=['test', 'test.*', 'test_*.py']),
    long_description=open('README.md', encoding='utf8').read(),
    long_description_content_type='text/markdown',
    author = 'lintao',
    author_email = 'lintao209@outlook.com',
    license='MIT',
    python_requires='>=3.6',

    install_requires=[
        'gym',
        'numpy',
        'torch'
    ],

    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],
)