import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="HEIP",
    version="0.0.1",
    author="Valeria Ariotta & Oskari Lehtonen",
    author_email="valeria.ariotta@helsinki.fi",
    description="End-to-end HE image analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://pypi.org/project/sfo/",
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 0 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: C',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        "Operating System :: OS Independent",
    ],
)
