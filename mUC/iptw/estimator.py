import pandas as pd
import numpy as np
import logging
from typing import List, Optional

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

logging.basicConfig(
    level = logging.INFO,                                 
    format = '%(asctime)s - %(levelname)s - %(message)s'  
)

class IPTWEstimator:
    
    def __init__(self):
        self.weights_df = None

    def weights(self, 
                df: pd.DataFrame, 
                treatment_col: str, 
                cat_var: Optional[List[str]] = None, 
                cont_var: Optional[List[str]] = None, 
                passthrough_var: Optional[List[str]] = None,
                stabilized: bool = False,
                lr_kwargs: Optional[dict] = None) -> Optional[pd.DataFrame]:
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
            List of column names to be treated as continuous variables. Missing values will be imputed with median 
            and all values will be standardzied using sklearn's StandardScaler. 
        passthrough_var : list of str, optional 
            Variables to be included without transformation. Typically used for binary indicators with no missing values.
        stabilized : bool, default = False
            If True, returns stabilized weights (multiplying IPTW by marginal probability of treatment or control).
        lr_kwargs : dict, optional
            Keyword arguments to pass directly to scikit-learn's LogisticRegression class.
            Common options include:
                - 'class_weight' : str (e.g., 'balanced'), None, or dict 
                - 'penalty' : 'l1', 'l2', 'elasticnet', or 'none'
                - 'solver' : 'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'
                - 'max_iter' : int (e.g., 1000)

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
            - All variables listed in cat_var must be of category dtype and contain no missing values. 
            - All variables listed in cont_var must be numeric dtype (int or float). 
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
            self.cat_var = cat_var or []
            self.cont_var = cont_var or []
            self.passthrough_var = passthrough_var or []
            
            df = df.copy()
            
            # Build pipeline
            numeric_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy = 'median')),
                ('scaler', StandardScaler())
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
            if lr_kwargs is None:
                lr_kwargs = {}
            lr_model = LogisticRegression(**lr_kwargs)
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

            self.weights_df = df
            return df

        except Exception as e:
            logging.error("Unable to calculate weights", exc_info=True)
            return None
        
    def plot_smd(self,
                 return_df: bool = False):
        """
        Compute and plots standardized mean differences (SMDs) before and after weighting for all variables used in the IPTW model.

        Parameters
        ----------
        return_df : bool, default = False
            If True, returns the DataFrame of SMD values along with the plot figure.

        This method uses internal attributes set during the `.weights()` call:
            - self.weights_df : the DataFrame with variables, treatment, and weights
            - self.cat_var : list of categorical variables
            - self.cont_var : list of continuous variables
            - self.passthrough_var : list of binary variables

        Returns
        -------
        matplotlib.figure.Figure
        A plot showing unweighted and weighted SMDs for all included variables.

        pd.DataFrame (optional)
            Returned only if `return_df=True`. Contains:
                - variable : covariate name
                - smd_unweighted : SMD with no weights
                - smd_weighted : SMD using IPTW weights

        Notes
        -----
        - SMD for continuous variables is calculated using pooled standard deviation (Cohen's d without degrees of freedom correction). 
            Median imputation is performed for missing values prior to calculating SMD. 
        - Categorical variables are one-hot-encoded prior to calculating SMD. 
        """
        # Input validation 
        if self.weights_df is None:
            raise ValueError("No weights found. Please run `.weights()` first.")
        
        all_var = self.cat_var + self.cont_var + self.passthrough_var
        if not all_var:
            raise ValueError("No variables found. Please run `.weights()` first.")
        
        try: 
            smd_df = self.weights_df[all_var + ['treatment', 'weight']].copy()
            treat = smd_df[smd_df['treatment'] == 1]
            control = smd_df[smd_df['treatment'] == 0]

            # Calculate SMD for continuous variables 
            # Impute median for cont_var
            for var in self.cont_var:
                smd_df[var] = smd_df[var].fillna(smd_df[var].median())

            smd_cont = []

            for var in self.cont_var:
                m1 = treat[var].mean()
                m0 = control[var].mean()
                s1 = treat[var].std()
                s0 = control[var].std()

                pooled_sd = np.sqrt(0.5 * (s1**2 + s0**2))
                smd_unweighted = (m1 - m0) / pooled_sd if pooled_sd > 0 else 0.0 # the if clause is a safety check to avoid dividing by zero

                m1 = np.average(treat[var], weights = treat['weight'])
                m0 = np.average(control[var], weights = control['weight'])
                s1 = np.sqrt(np.average((treat[var] - m1) ** 2, weights = treat['weight']))
                s0 = np.sqrt(np.average((control[var] - m0) ** 2, weights = control['weight']))

                pooled_sd = np.sqrt(0.5 * (s1**2 + s0**2))
                smd_weighed = (m1 - m0) / pooled_sd if pooled_sd > 0 else 0.0 # the if clause is a safety check to avoid dividing by zero

                smd_cont.append({
                    'variable': var,
                    'smd_unweighted': smd_unweighted,
                    'smd_weighted': smd_weighed
                })

            # Return dataframe if requested
            if return_df:
                return fig, smd_df

            return fig
            
        except: 
            logging.error("Unable to calculate and plot SMD", exc_info=True)
            return None
        
        


    