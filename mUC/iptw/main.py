import pandas as pd
import numpy as np
import logging
from typing import List, Optional

from estimator import IPTWEstimator

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

from IPython import embed

if __name__ == "__main__":
    estimator = IPTWEstimator()

    dtype_map = pd.read_csv('../outputs/final_df_dtypes.csv', index_col = 0).iloc[:, 0].to_dict()
    df = pd.read_csv('../outputs/final_df.csv', dtype = dtype_map)
    df_death = pd.read_csv('../outputs/full_cohort_with_death_data.csv', dtype = dtype_map)
    df = pd.merge(df, df_death[['PatientID', 'LineName']], on = 'PatientID', how = 'left')
    df['treatment'] = np.where(df['LineName'] == 'chemo', 0, 1)


    a = estimator.fit_transform(df = df,
                  treatment_col='treatment',
                  cat_var = ['ecog_index', 'GroupStage_mod'],
                  cont_var = ['age', 'creatinine', 'albumin'],
                  binary_var= ['Surgery', 'ecog_newly_gte2'],
                  stabilized=False)
    
    #a.to_csv('../outputs/iptw_df.csv', index = False)

    b, c = estimator.smd(return_fig = True)
    #b.to_csv('../outputs/smd_df.csv', index = False)

    embed()
