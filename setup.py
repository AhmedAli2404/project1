from setuptools import setup,find_packages


HYPEN_E_DOT='-e .'

def get_packages(file_path):
    required_pacakges=[]
    with open(file_path) as file:
        reader1=file.readlines()
        required_pacakges=[req.replace("\n","") for req in reader1]


        if HYPEN_E_DOT in required_pacakges:
            required_pacakges.remove(HYPEN_E_DOT)

    return required_pacakges







setup(
    name="LinearRegression",
    version="0.0.1",
    author_name="Ahmed Ali",
    author_email="pathanahmedali786@gmail.com",
    install_requires=get_packages("requirements.txt"),
    packages=find_packages()
)