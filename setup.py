import setuptools

version = "0.0.1"

with open("model_trust/version.py", "w") as f:
    f.write('# generated by setup.py\nversion = "{}"\n'.format(version))


install_requires = [
    "numpy>=1.23.1",
    "pandas>=1.4.3",
    "lightgbm>=4.0.0",
    "scikit-learn==1.1.1",
    "torch>=2.0.1",
    "protobuf==3.20.2",
    "optuna==3.4.0",
    "mlprodict",
    "jyquickhelper",
    "skl2onnx",
    "onnxruntime",
]

extra_requires = {"plot": ["matplotlib"]}

# pip install --no-binary lightgbm --config-settings=cmake.define.USE_OPENMP=OFF 'lightgbm>=4.0.0'


setuptools.setup(
    name="model_trust",
    version=version,
    description="IBM Model Trust",
    authos="IBM Research",
    url="https://github.com/trustyai-explainability/trustyai-model-trust",
    author_email="Natalia.Martinez.Gil@ibm.com, j.srideepika@ibm.com, giridhar.ganapavarapu@ibm.com",
    packages=setuptools.find_packages(),
    license="Apache License 2.0",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    extras_require={
        "test": [
            "pytest",
        ]
    },
    package_data={},
    include_package_data=True,
    zip_safe=False,
)
