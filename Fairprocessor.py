import numpy as np
import copy
import random
import pandas as pd
import time

# ML Learners
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

#Function to find the K nearest neighhours
def get_ngbr(df, knn):
    #np.random.seed(0)
    rand_sample_idx = random.randint(0, df.shape[0] - 1)
    parent_candidate = df.iloc[rand_sample_idx]
    distance,ngbr = knn.kneighbors(parent_candidate.values.reshape(1,-1),3,return_distance=True)    
    candidate_1 = df.iloc[ngbr[0][1]]    
    return parent_candidate,candidate_1

def fair_processor_samples(no_of_samples,df,df_name,X_train,y_train,protected_attribute):

    k=0 #Counter number for creating data instance
    #----------------------------------------------------------------------------------------------
    #Calling function to find the KNN
   
    total_data = df.values.tolist()
    knn = NN(n_neighbors=5,algorithm='auto').fit(df)
     #-----------------------------------------------------------------------------------------
    #Make a complete DataFrame
    X_train['Probability']=y_train
    column_name=X_train.columns.tolist()
    #--------------------------------------------------------------------------------------------------------------
    #Logic to create synthetic data

    while(k!=no_of_samples):
        
        f = .80 #cross over frequency
        parent_candidate, child_candidate_1 = get_ngbr(df, knn)      
        mutant = [] #to store new mutants
        for key,value in parent_candidate.items():    
       
            #For boolean cases
            if isinstance(parent_candidate[key], bool):
                mutant.append(np.random.choice([parent_candidate[key],child_candidate_1[key]])) 
                    #print('string: x1 less than x2')
             
            #For string cases        
            elif isinstance(parent_candidate[key], str):
                mutant.append(np.random.choice([parent_candidate[key],child_candidate_1[key]]))   
                    #print('string: x1 less than x2')

            #For numeric cases     
            else:             
                mutant.append(parent_candidate[key] + f * (child_candidate_1[key] - parent_candidate[key]))
                    #print('integer: x1 less than x2')  
            k=k+1
               
        #--------------------------------------------------------------------------------------------------------------


    
    
    final_df = pd.DataFrame(total_data)

    #--------------------------------------------------------------------------------------------------------------
    #Rename dataframe columns
    final_df.set_axis(column_name, axis=1,inplace=True)

    return final_df