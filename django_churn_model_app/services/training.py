import os 
import pandas as pd 
import numpy as np 
from rest_framework import status 
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,recall_score,accuracy_score,precision_score,confusion_matrix,classification_report
import constant as AppConst 
from sklearn.model_selection import train_test_split
import pickle

base_path = os.getcwd() 
data_path=os.path.normpath(base_path+os.sep+'data')
pickle_path=os.path.normpath(base_path+os.sep+'pickle')
log_path=os.path.normpath(base_path+os.sep+'log')

class Training: 

    def train(self,request): 
        return_dict=dict()
        try: 
            train_data =os.path.normpath(data_path+os.sep+'train_data.csv')
            df=pd.read_csv(train_data)
            df=df.fillna(0)
            X,y=self.get_feat_and_target(df,AppConst.TARGET)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42,stratify=y)
            model = RandomForestClassifier(max_depth=AppConst.MAX_DEPTH,n_estimators=AppConst.N_ESTIMATORS)
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            accuracy,precision,recall,f1score = self.accuracymeasures(y_test,y_pred,'weighted')
            
            
            pickle_file = os.path.normpath(pickle_path+os.sep+'model.sav')
            pickle.dump(model, open(pickle_file, 'wb'))

            return_dict['response'] = 'Model Trained Successfully'
            return_dict['status']=status.HTTP_200_OK
            return return_dict 

        except Exception as e: 
            return_dict['response']="Exception when training the module: "+str(e.__str__)
            return_dict['status']=status.HTTP_500_INTERNAL_SERVER_ERROR
            return return_dict 


    def accuracymeasures(self,y_test,predictions,avg_method):
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average=avg_method)
        recall = recall_score(y_test, predictions, average=avg_method)
        f1score = f1_score(y_test, predictions, average=avg_method)
        target_names = ['0','1']
        print("Classification report")
        print("---------------------","\n")
        print(classification_report(y_test, predictions,target_names=target_names),"\n")
        print("Confusion Matrix")
        print("---------------------","\n")
        print(confusion_matrix(y_test, predictions),"\n")

        print("Accuracy Measures")
        print("---------------------","\n")
        print("Accuracy: ", accuracy)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1 Score: ", f1score)

        return accuracy,precision,recall,f1score

    def get_feat_and_target(self,df,target):
        """
        Get features and target variables seperately from given dataframe and target 
        input: dataframe and target column
        output: two dataframes for x and y 
        """
        x=df.drop(target,axis=1)
        y=df[[target]]
        return x,y    

    