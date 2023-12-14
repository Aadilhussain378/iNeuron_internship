from setuptools import find_packages,setup

setup(
    name='TravelPackagePrediction',
    version='0.0.1',
    author='Aadil Hussain',
    author_email='hussainaadil378@gmail.com',
    install_requires=["scikit-learn","pandas","numpy"],
    packages=find_packages()
)