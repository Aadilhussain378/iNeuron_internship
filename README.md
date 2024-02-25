## TRAVEL PACKAGE PURCHASE PREDICTION
An End to  End project using CICD pipeline which will help us to predict whether a new Customer will Purchase the Package or Not.
***
## General info
The "Travel Purchase Package Prediction" project revolves around a binary classification problem within the realms of travel and data science. Specifically designed to forecast travel package purchases, the system harnesses historical data encompassing customer interactions and purchase behaviors. The primary goal is to categorize customers into two distinct groups: those likely to purchase the travel package and those not likely to do so.
Drawing valuable insights from past purchase records and various customer attributes, the system employs a diverse range of machine learning algorithms. These algorithms meticulously identify intricate patterns and relationships within the data, contributing to the development of a robust binary classification model. The model's efficacy lies in its ability to accurately classify customers, providing businesses with insights to tailor marketing strategies and optimize sales efforts.
In essence, this project revolutionizes customer engagement and revenue generation in the travel industry by empowering businesses to proactively address the diverse needs and preferences of their customer base. The outcome is a sophisticated binary classification model that enhances the precision of predictions, serving as a strategic tool for businesses seeking to elevate their marketing initiatives and maximize sales in the dynamic landscape of the travel sector.
***
## ⏳Data Source
[Data source:] (https://question.transtutors.com/6129343_1_tourism-data.xlsx)
Train and Test data are stored in .csv format.
***
## Problem Statement
In the fast-growing world of tourism, predicting whether people will buy travel packages has become really important. The tourism industry can have unpredictable changes in demand, so being able to accurately predict if customers will purchase travel packages helps organizations plan better. The main goal here is to figure out if a customer is likely to buy a travel package or not. This information is valuable for tourism businesses to make smart decisions and adapt to the changing needs of travelers.
***
## 📷 Demo Photos
<img width="800" alt="Page1" src="https://github.com/Aadilhussain378/iNeuron_internship/assets/112958214/8b4a6e43-3783-48bc-921e-d465b062c721">
<img width="800" alt="Page1" src="https://github.com/Aadilhussain378/iNeuron_internship/assets/112958214/3ce50d35-f42e-4946-8f1e-30d2b31838b7">

This interface functions as the portal to our project. Ensure accurate input across all fields, and the system will generate predictions tailored to your specifications. Specifically, it will predict whether the customer is likely to purchase the package or not based on the information you provide. Your detailed inputs enable the system to deliver precise and personalized forecasts, aiding in strategic decision-making for travel package offerings.
***
## Library Used
A file named "requirements.txt" will include a list of essential libraries utilized in the project.
 ```
pandas
numpy
seaborn
matplotlib
scikit-learn
SVC
flask
dill
 ```
## ⚙️Project Structure
 ### We have used the following structure to develop this End-To-End Project:
 * ```setup.py```It is a script in Python that is used for packaging and distributing Python projects.
 * ```requirements.txt``` it will have all the packages that i really need to install while im implementing the project.
 * ```logger.py``` It helps in capturing detailed information about events occurring at specific times or within specific files.
 * ```exception.py``` takes charge of handling custom exceptions that arise when an error occurs in any file. It provides detailed information including the file name, line number, and the nature of the error.
 * ```.gitignore``` prevents the inclusion of specific files that we don't want to push to GitHub.
 * ```readme.md``` contain general informtion about the project steps and requiremnts for further explaination.
 * ```data```contain the dataset.
 * ```src``` contain many subfolder. we need to give a ```__init__.py``` file in each directory so that we can use each file as a package.
 * ```src/data_ingestion.py``` It is a part of a module when we are developing a project. it will have all the code that will be related to reading the data.
 * ```src/data_transformation.py```After ingesting the data i may do transformation of data or validation of data. For this we will have this file. Over here will be probably writing the code, how to change the categorical variable to numerical variable, how to handle the one hot encoding or label encoding etc.
 * ```src/Model_trainer.py```handles both model training and hyperparameter tuning. It returns a model pickle file that is trained on the provided data and can be utilized for subsequent predictions.
 * ```src/Prediction_Pipeline.py``` is responsible for the Creating the Pipeline using the ```app.py```
 * ```utils.py``` is used for creating and storing the common function which are used through out the Project.
 * ```app.py``` serves as the web application file that interacts with users. 
 

## Run Locally
Here I will tell you, how you can run this project locally.
* For this make sure you have git, Anaconda or miniconda installed on your system
* Clone the complete project with git clone https://github.com/Aadilhussain378/iNeuron_internship.git

* Once the project is cloned, open the terminal in vscode and run this command for creatinng the environment. ```conda create -p venv python==3.8 -y``` after that you need activate the environment by ruuning the command as:  ```source activate venv/.```
* Subsequently, execute the command ```pip install -r requirements.txt``` to install all the necessary dependencies. Conclude by launching the project using the command ```python app.py```. Open the provided localhost URL, and you can now interact with the project effortlessly by either opening your web browser or entering ```http://localhost:8080/predictdata``` in the address bar.

## Deployment Techniques
* Deployment to AWS: The final step is to deploy the All files to an AWS server. This step is done by us using the ubuntu server where we deployed our model using the winSCP to connect the AWS server(EC2).

## 🎯Project Created BY
[@AadilHuusain](https://www.linkedin.com/in/aadilhussain378/)