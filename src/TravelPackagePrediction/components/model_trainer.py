import os
import sys
from dataclasses import dataclass


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.TravelPackagePrediction.exception import CustomException
from src.TravelPackagePrediction.logger import logging

from src.TravelPackagePrediction.utils.utils import save_object,evaluate_models



@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.Model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting Dependent and Independent variables from train and test data")
            
            
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            

            models = {
                "LogisticRegression":LogisticRegression(),
                "RandomForestClassifier":RandomForestClassifier(),
                "NaiveBayesClassifier":GaussianNB(),
                "DecisionTreeClassifier":DecisionTreeClassifier(),
                "SupportVectorMachine":SVC(),
                "KNeighborsClassifier":KNeighborsClassifier(),
                "AdaBoostClassifier":AdaBoostClassifier(),
            }

            params={
                    "RandomForestClassifier":{
                    'criterion':['gini', 'log_loss', 'entropy'],
                    'max_features':['sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256] },

                    "DecisionTreeClassifier":{
                    'criterion':['entropy', 'gini', 'log_loss'],
                                 'splitter':['best','random'],
                                'max_features':['sqrt','log2']},

                    "SupportVectorMachine":{
                            'C': [0.1, 1, 10, 100, 1000], 
                            'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                            'kernel': ['rbf']},
                        
                        "KNeighborsClassifier":{'n_neighbors' : [5,7,9,11,13,15],
                            'weights' : ['uniform','distance'],
                            'metric' : ['minkowski','euclidean','manhattan']},
                        
                        
                        
                         "AdaBoostClassifier":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]},
                        
                     "LogisticRegression":{},
                        
                     "NaiveBayesClassifier":{}
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## Get the best model score from the dict
            best_model_score=max(sorted(model_report.values()))

            ## To get the best model name from dict
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            


            best_model=models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            print(f'Best Model Found , Model Name : {best_model} , accuracy : {best_model_score}')
            print('\n====================================================================================\n')

            logging.info("Best model found on both the training and testing data set")
            save_object(
                file_path=self.Model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            
            
            
        except Exception as e:
            raise CustomException(e,sys)