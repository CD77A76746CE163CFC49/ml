import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import pickle, re, os, gensim
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.impute import SimpleImputer
import category_encoders as ce
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


def rmse(y_test,y_pred):      
    return np.sqrt(mean_squared_error(y_test,y_pred))

def main():

    train = pd.read_csv("tcd ml 2019-20 income prediction training (with labels).csv" ,index_col = 'Instance')
 
    train.dropna()
    train.reset_index(inplace = False)

    train.drop(['Wears Glasses', 'Hair Color'], axis=1,inplace=True)

    # Remove Outliers
      
    threshold = 3
    train_year_mean = np.mean(train['Year of Record'])
    train_year_sd = np.std(train['Year of Record'])
    train_year_upper = train_year_mean + threshold*train_year_sd
    train_year_lower = train_year_mean - threshold*train_year_sd
    train = train[(train['Year of Record'] <train_year_upper) & 
                   (train['Year of Record'] >train_year_lower)]
    
    train_age_mean = np.mean(train['Age'])
    train_age_sd = np.std(train['Age'])
    train_age_upper = train_age_mean + threshold*train_age_sd
    train_age_lower = train_age_mean - threshold*train_age_sd
    train = train[(train['Age'] <train_age_upper) & 
                   (train['Age'] >train_age_lower)]

    train_citysize_mean = np.mean(train['Size of City'])
    train_citysize_sd = np.std(train['Size of City'])
    train_citysize_upper = train_citysize_mean + threshold*train_citysize_sd
    train_citysize_lower = train_citysize_mean - threshold*train_citysize_sd
    train = train[(train['Size of City'] <train_citysize_upper) & 
                   (train['Size of City'] >train_citysize_lower)]
    
    train_bodyheight_mean = np.mean(train['Body Height [cm]'])
    train_bodyheight_sd = np.std(train['Body Height [cm]'])
    train_bodyheight_upper = train_bodyheight_mean + threshold*train_bodyheight_sd
    train_bodyheight_lower = train_bodyheight_mean - threshold*train_bodyheight_sd
    train = train[(train['Body Height [cm]'] <train_bodyheight_upper) & 
                   (train['Body Height [cm]'] >train_bodyheight_lower)]





    train = train[(train['Income in EUR'].values > 0) & (train['Income in EUR'].values < 1000000)]
    train.reset_index(inplace = True)

    gender_list = ['male','female','other','unknown']
    train['Gender'][~train.Gender.isin(gender_list)] = 'unknown'

    degree_list = ['Bachelor','Master','Phd','No']
    train['University Degree'][~train['University Degree'].isin(degree_list)] = 'unknown'

    train_impute = SimpleImputer(strategy="most_frequent")
    train = pd.DataFrame(data = train_impute.fit_transform(train), columns=train.columns)

    train["PopulationCatg"] = np.where(train['Size of City'] < 900000 , "High", "Low")
    train.drop(['Size of City'], axis=1, inplace=True)

    train.to_csv("train_impute.csv",sep=",", index=False)

    #Change Float to Int for "Year"
    train['Year of Record'] = train['Year of Record'].astype(int)
    train['Age'] = train['Age'].astype(int)
    train.dropna(inplace=True)
    train['Body Height [cm]'] = train['Body Height [cm]'].astype(int)
    train.dropna(inplace=True)
    train.reset_index(inplace=True, drop=True)


#######

    train_sub = train[['Year of Record','Age','Body Height [cm]']]


    train.drop(['Year of Record', 'Age','Body Height [cm]'],axis=1, inplace=True)

    train_scaler = MinMaxScaler()
    train_scaled = pd.DataFrame(train_scaler.fit_transform(train_sub),
                                columns=['Year of Record', 'Age','Body Height [cm]'])


    train_columns = np.concatenate([train.columns.values, train_scaled.columns.values], axis= 0)
    train = pd.DataFrame(np.hstack([train, train_scaled]),columns=train_columns)

    y = train.loc[:, ['Income in EUR']].astype('int')
    X = train.copy()
    X.drop(['Instance','Income in EUR'],axis=1, inplace=True)



"""
    Encoding
"""

    encoder = ce.TargetEncoder(cols=['Country','Year of Record','Gender','University Degree','PopulationCatg','Profession'])
    encoder.fit(X[['Country','Year of Record','Gender','University Degree','PopulationCatg','Profession']], y)
    X_cleaned = encoder.transform(X[['Country','Year of Record','Gender','University Degree','PopulationCatg','Profession']], y)
    X.drop(['Country','Year of Record','Gender','University Degree','PopulationCatg','Profession'], axis =1, inplace = True)
    X_columns = np.concatenate([X.columns.values, X_cleaned.columns.values], axis= 0)
    X = pd.DataFrame(np.hstack([X, X_cleaned]),columns=X_columns)


   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.37, random_state=1)
#-----------------------------------------------------------------------------------------------
   
#-----------------------------------------------------------------------------------------------    

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1500, n_jobs = -1 )# Train the model on training data
rf.fit(X_train, y_train)

# Use the forest's predict method on the test data
x_predictions = rf.predict(X_train)

    
    y_test_pred = rf.predict(X_test)

    train_mse2 = mean_squared_error(x_predictions, y_train)
    test_mse2 = mean_squared_error(y_test_pred, y_test)
    train_rmse2 = np.sqrt(train_mse2)
    test_rmse2 = np.sqrt(test_mse2)
    print('Train RMSE: %.4f' % train_rmse2)
    print('Test RMSE: %.4f' % test_rmse2)
    
    
    import pickle
    
    pickle.dump(rf, open("Salary_RandomForestRegressor_Model.pickle.dat", "wb"))

#-----------------------------------------------------------------------------------------------
    
    # Prediction

    test = pd.read_csv("tcd ml 2019-20 income prediction test (without labels).csv")

    test.drop(['Wears Glasses', 'Hair Color','Income'],axis=1, inplace=True)

    gender_list = ['male','female','other','unknown']
    test['Gender'][~test.Gender.isin(gender_list)] = 'unknown'

    degree_list = ['Bachelor','Master','Phd','No']
    test['University Degree'][~test['University Degree'].isin(degree_list)] = 'unknown'

    test_impute = SimpleImputer(strategy="most_frequent")
    test = pd.DataFrame(data = test_impute.fit_transform(test), columns=test.columns)
    
    test["PopulationCatg"] = np.where(test['Size of City'] < 900000 , "High", "Low")
    test.drop(['Size of City'], axis=1, inplace=True)

    test['Year of Record'] = test['Year of Record'].astype(int)
    test['Age'] = test['Age'].astype(int)
    test.dropna(inplace=True)
    test['Body Height [cm]'] = test['Body Height [cm]'].astype(int)
    test.dropna(inplace=True)

    test.reset_index(inplace=True, drop=True)

    test_sub = test.loc[:, ['Year of Record','Age','Body Height [cm]']]

     test.drop(['Year of Record','Age', 'Body Height [cm]'], axis=1, inplace=True)

    test_scaler = MinMaxScaler()

    test_scaled = pd.DataFrame(test_scaler.fit_transform(test_sub), columns=['Year of Record','Age', 'Body Height [cm]'])

    test_columns = np.concatenate([test.columns.values, test_scaled.columns.values], axis= 0)
    test = pd.DataFrame(np.hstack([test, test_scaled]),columns=test_columns)

    X_test = test.copy()
    X_test.drop(['Instance'], axis=1, inplace=True)

    X_test_cleaned = encoder.transform(X_test[['Country','Year of Record','Gender','University Degree','PopulationCatg','Profession']])

    X_test.drop(['Country','Year of Record','Gender','University Degree','PopulationCatg','Profession'], axis =1, inplace = True)

    X_test_columns = np.concatenate([X_test.columns.values, X_test_cleaned.columns.values], axis= 0)
    X_test = pd.DataFrame(np.hstack([X_test, X_test_cleaned]),columns=X_test_columns)


    # load model from file
    loaded_model = pickle.load(open("Salary_RandomForestRegressor_Model.pickle.dat", "rb"))

    y_pred = loaded_model.predict(X_test)



    test['Income'] = y_pred

    test[['Instance','Income']].to_csv(path_or_buf  ="tcd ml 2019-20 income prediction submission file.csv",
                                    sep=",",header=True,index=False)
