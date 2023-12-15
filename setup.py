from setuptools import find_packages,setup

"""HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements"""

setup(
    name='TravelPackagePrediction',
    version='0.0.1',
    author='Aadil Hussain',
    author_email='hussainaadil378@gmail.com',
    packages=find_packages(),
    install_requires=("pandas","numpy","seaborn","matplotlib","scikit-learn")
)