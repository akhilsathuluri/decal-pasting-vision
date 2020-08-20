import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="remote-plc-diagnostics",
    version="0.0.1",
    author="Akhil Sathuluri",
    author_email="example@example.com",
    description="An IIoT package for remote PLC control and diagnostics via your local network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/akhilsathuluri/remote",
    packages=setuptools.find_packages(),
    install_requires=[
          # 'sqlite',
          'pymodbus',
          'streamlit==0.62.1', # Currently any version above this has pyarrow error on RPi
          'pandas',
          'sqlalchemy',
          'xlrd'
      ],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
