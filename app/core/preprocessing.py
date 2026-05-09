import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# RowNumber,CustomerId,Surname,CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Exited
class Churn_Modelling:
    def __init__(self):
        self.numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary','HasCrCard','IsActiveMember']
        self.categorical_features = ['Geography', 'Gender'] 
    
        num_tranformer= Pipeline(steps=
           [('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())])
        cat_transformer=Pipeline(steps=[('imputer',SimpleImputer(strategy='constant', missing_values=' ',fill_value='missing')),('OneHotEncoder',OneHotEncoder(handle_unknown='ignore'))])

        self.preprocessor=ColumnTransformer(transformers=[('num',num_tranformer,self.numeric_features),('cat',cat_transformer,self.categorical_features)])

    def fit_transform(self,df):
        df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, errors='ignore')
        return self.preprocessor.fit_transform(df)

    def transform(self,df):
        df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, errors='ignore')
        return self.preprocessor.transform(df)
    

    
