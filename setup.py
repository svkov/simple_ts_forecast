import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="simple-ts-forecast",
    version="0.0.1",
    author="svkov",
    author_email="kovalev.svyatoslav42@gmail.com",
    description="A simle library for time series forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/svkov/simple_ts_forecast",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'pandas',
        'statsmodels',
        'pmdarima',
        'matplotlib',
        'pywt',
        'sklearn'
    ],
    python_requires='>=3.8',
)
