import pandas as pd
import numpy as np
import warnings
from numpy import random
warnings.filterwarnings('ignore')
import sys

def get_data():
    try:
        data= pd.read_csv("covid_data/data.csv")
    except FileNotFoundError:
         sys.exit('Please, download the covid data from https://www.kaggle.com/datasets/meirnizri/covid19-dataset and put the file (renamed to "data.csv") into the covid_data folder.')
    df=data.copy()
    df.DATE_DIED[df['DATE_DIED'] != '9999-99-99'] = 1
    df.DATE_DIED[df['DATE_DIED'] == '9999-99-99'] = 0
    df['DEATH'] = df.DATE_DIED
    df.drop(columns=["DATE_DIED"], inplace=True)
    df['PREGNANT'].replace({97 : 0, 98 : np.nan}, inplace = True)
    df['USMER'].replace(2.0, 0, inplace=True)
    df['SEX'].replace(2.0, 0, inplace=True)
    df['PATIENT_TYPE'].replace(2.0, 0, inplace=True)
    df['INTUBED'].replace(2.0, 0, inplace=True)
    df['PNEUMONIA'].replace(2.0, 0, inplace=True)
    df['PREGNANT'].replace(2.0, 0, inplace=True)
    df['DIABETES'].replace(2.0, 0, inplace=True)
    df['COPD'].replace(2.0, 0, inplace=True)
    df['ASTHMA'].replace(2.0, 0, inplace=True)
    df['INMSUPR'].replace(2.0, 0, inplace=True)
    df['HIPERTENSION'].replace(2.0, 0, inplace=True)
    df['OTHER_DISEASE'].replace(2.0, 0, inplace=True)
    df['CARDIOVASCULAR'].replace(2.0, 0, inplace=True)
    df['OBESITY'].replace(2.0, 0, inplace=True)
    df['RENAL_CHRONIC'].replace(2.0, 0, inplace=True)
    df['TOBACCO'].replace(2.0, 0, inplace=True)
    df['ICU'].replace(2.0, 0, inplace=True)
    df.replace([97,98, 99], np.nan, inplace = True)

    df.drop(columns=["INTUBED","ICU"], inplace=True)

    fill_list = df['PNEUMONIA'].dropna()
    df['PNEUMONIAe'] = df['PNEUMONIA'].fillna(pd.Series(np.random.choice(fill_list , size = len(df.index))))

    fill_list = df['AGE'].dropna()
    df['AGE'] = df['AGE'].fillna(pd.Series(np.random.choice(fill_list , size = len(df.index))))

    fill_list = df['PREGNANT'].dropna()
    df['PREGNANT'] = df['PREGNANT'].fillna(pd.Series(np.random.choice(fill_list , size = len(df.index))))

    fill_list = df['DIABETES'].dropna()
    df['DIABETES'] = df['DIABETES'].fillna(pd.Series(np.random.choice(fill_list , size = len(df.index))))

    fill_list = df['COPD'].dropna()
    df['COPD'] = df['COPD'].fillna(pd.Series(np.random.choice(fill_list , size = len(df.index))))

    fill_list = df['ASTHMA'].dropna()
    df['ASTHMA'] = df['ASTHMA'].fillna(pd.Series(np.random.choice(fill_list , size = len(df.index))))

    fill_list = df['INMSUPR'].dropna()
    df['INMSUPR'] = df['INMSUPR'].fillna(pd.Series(np.random.choice(fill_list , size = len(df.index))))

    fill_list = df['HIPERTENSION'].dropna()
    df['HIPERTENSION'] = df['HIPERTENSION'].fillna(pd.Series(np.random.choice(fill_list , size = len(df.index))))

    fill_list = df['OTHER_DISEASE'].dropna()
    df['OTHER_DISEASE'] = df['OTHER_DISEASE'].fillna(pd.Series(np.random.choice(fill_list , size = len(df.index))))

    fill_list = df['CARDIOVASCULAR'].dropna()
    df['CARDIOVASCULAR'] = df['CARDIOVASCULAR'].fillna(pd.Series(np.random.choice(fill_list , size = len(df.index))))

    fill_list = df['OBESITY'].dropna()
    df['OBESITY'] = df['OBESITY'].fillna(pd.Series(np.random.choice(fill_list , size = len(df.index))))

    fill_list = df['RENAL_CHRONIC'].dropna()
    df['RENAL_CHRONIC'] = df['RENAL_CHRONIC'].fillna(pd.Series(np.random.choice(fill_list , size = len(df.index))))

    fill_list = df['TOBACCO'].dropna()
    df['TOBACCO'] = df['TOBACCO'].fillna(pd.Series(np.random.choice(fill_list , size = len(df.index))))

    df.dropna(subset=['PNEUMONIA'],inplace=True)
    return np.append(df.to_numpy(), np.array([[i] for i in random.randint(10000, size=len(df.index))]) , axis=1)
    