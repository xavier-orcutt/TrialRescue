import pandas as pd
import numpy as np
import logging
from typing import List, Optional

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

logging.basicConfig(
    level = logging.INFO,                                 
    format = '%(asctime)s - %(levelname)s - %(message)s'  
)

def weights(df: pd.DataFrame, 
            treatment_col: str, 
            cat_var: Optional[List[str]] = None, 
            cont_var: Optional[List[str]] = None, 
            passthrough_var: Optional[List[str]] = None,
            stabilized: bool = False) -> Optional[pd.DataFrame]:
    """
    Calculate inverse probability of treatment weights (IPTW) using logistic regression.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing treatment assignment and variables of interest for calculating weights.
    treatment_col : str
        Name of the column indicating treatment assignment (binary: 0 or 1).
    cat_var : list of str, optional
        List of column names to be treated as categorical variables. These will be one-hot encoded.
    cont_var : list of str, optional 
        List of column names to be treated as continuous variables. Missing values will be imputed with median. 
    passthrough_var : list of str, optional 
        List of variables to be included in the model without any transformation (e.g., binary variables without missingness)
    stabilized : bool, default = False
        If True, returns stabilized weights (multiplying IPTW by marginal probability of treatment or control).

    Returns
    -------
    pd.DataFrame or None
       A copy of the original DataFrame with the following columns: 
            'propensity_score' : float
                calculated propensity scores 
            'weight' : float 
                calculated inverse probability of treatment weight 
       
       Returns None if insufficient input is provided.

    Notes
    -----
    - This function uses logistic regression to estimate the propensity score `P(Treatment=1 | X)`.
    - Assumes binary treatment coded as 0 (control) and 1 (treated).
    
    - Regarding inputted variables: 
        - At least one of `cat_var`, `cont_var`, or `passthrough_var` must be provided; otherwise, the function will not proceed.
        - All variables listed in cat_var must be of pandas category dtype and contain no missing values. 
        - All variables listed in cont_var must be numeric (int or float dtypes). 
        - All variables listed in passthrough_var must contain no missing values. 
    """

    # Input validation 
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    
    if treatment_col not in df.columns:
        raise ValueError('treatment_col not found in df')
    if not set(df[treatment_col].unique()).issubset({0, 1}):
        raise ValueError('treatment_col must contain only binary values (0 and 1)')
    if df[treatment_col].isnull().any():
        raise ValueError('treatment_col has missing values')
    
    if all(var is None for var in [cat_var, cont_var, passthrough_var]):
        raise ValueError('at least one of cat_var, cont_var, or passthrough_var must be provided')
    
    if cat_var is not None:
        # Check that columns in cat_var are present in the df
        missing = [col for col in cat_var if col not in df.columns]
        if missing:
            raise ValueError(f"The following columns in cat_var are missing from the DataFrame: {missing}")
        
        # Check that columns in cat_var are categorical
        non_categorical = [col for col in cat_var if not pd.api.types.is_categorical_dtype(df[col])]
        if non_categorical:
            raise ValueError(f"The following columns in cat_var are not categorical dtype: {non_categorical}")

        # Check that columns in cat_var have no missing values 
        cat_missing = [col for col in cat_var if df[col].isnull().any()]
        if cat_missing:
            raise ValueError(f"The following columns in cat_var have missing values: {cat_missing}")

    if cont_var is not None:
        # Check that columns in cont_var are present in the df
        missing = [col for col in cont_var if col not in df.columns]
        if missing:
            raise ValueError(f"The following columns in cont_var are missing from the DataFrame: {missing}")
        
        # Check that columns in cont_var are numerical type 
        non_numeric = [col for col in cont_var if not pd.api.types.is_numeric_dtype(df[col])]
        if non_numeric:
            raise ValueError(f"The following columns in cont_var are not numeric: {non_numeric}")

    if passthrough_var is not None:
        # Check that columns in passthrough_var are present in the df
        missing = [col for col in passthrough_var if col not in df.columns]
        if missing:
            raise ValueError(f"The following columns in passthrough_var are missing from the DataFrame: {missing}")
        
        # Check that columns in passthrough_var have no missing values
        pt_missing = [col for col in passthrough_var if df[col].isnull().any()]
        if pt_missing:
            raise ValueError(f"The following columns in passthrough_var have missing values: {pt_missing}")

    try:    
        df = df.copy()
        
        # Build pipeline
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy = 'median')),
            ('scaler', RobustScaler())
        ])

        categorical_pipeline = Pipeline([
            ('encoder', OneHotEncoder(handle_unknown = 'ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers = [
                ('num', numeric_pipeline, cont_var),
                ('cat', categorical_pipeline, cat_var),
                ('pass', 'passthrough', passthrough_var)],
                remainder = 'drop'
        )

        # Fit and transform
        X_preprocessed = preprocessor.fit_transform(df)

        # Calculating propensity scores using logistic regression 
        lr_model = LogisticRegression(class_weight = 'balanced')
        lr_model.fit(X_preprocessed, df[treatment_col])
        propensity_score = lr_model.predict_proba(X_preprocessed)[:, 1] # Select second column for probability of receiving treatment 
        df['propensity_score'] = propensity_score
        
        # If stabilized == True, calculate stabilized weight
        if stabilized:
            p_treated = df[treatment_col].mean()
            df['weight'] = np.where(df[treatment_col] == 1,
                                    p_treated / df['propensity_score'],
                                    (1 - p_treated) / (1 - df['propensity_score']))
            
        # Otherwise, calculate unstabilized weights 
        else: 
            df['weight'] = np.where(df[treatment_col] == 1,
                                    1/df['propensity_score'], 
                                    1/(1 - df['propensity_score']))

        return df

    except Exception as e:
        logging.error("Unable to calculate weights", exc_info=True)
        return None
    
# TESTING
from IPython import embed

dtype_map = pd.read_csv('../outputs/final_df_dtypes.csv', index_col = 0).iloc[:, 0].to_dict()
df = pd.read_csv('../outputs/final_df.csv', dtype = dtype_map)
df_death = pd.read_csv('../outputs/full_cohort_with_death_data.csv', dtype = dtype_map)
df = pd.merge(df, df_death[['PatientID', 'LineName']], on = 'PatientID', how = 'left')
df['treatment'] = np.where(df['LineName'] == 'chemo', 0, 1)

a = weights(df = df,
            treatment_col='treatment',
            cat_var = ['ecog_index', 'GroupStage_mod'],
            cont_var = ['age', 'creatinine', 'albumin'],
            passthrough_var= ['Surgery', 'ecog_newly_gte2'],
            stabilized=True)

embed()