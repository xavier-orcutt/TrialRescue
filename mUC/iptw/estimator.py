import pandas as pd
import numpy as np
import logging
from typing import Optional, Union, Tuple, List

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
        self.treatment_col = None
        self.cat_var = []
        self.cont_var = []
        self.binary_var = []
        self.all_var = []
        self.stabilized = False
        self.clip_bounds = None

        self.propensity_score_df = None
        self.weight_df = None
        self.weight_col = None
        self.smd_results_df = None

    def fit(self, 
            df: pd.DataFrame, 
            treatment_col: str, 
            cat_var: Optional[List[str]] = None, 
            cont_var: Optional[List[str]] = None, 
            binary_var: Optional[List[str]] = None,
            stabilized: bool = False,
            lr_kwargs: Optional[dict] = None,
            clip_bounds: Optional[Union[Tuple[float, float], List[float]]] = None) -> None:
        """
        Fit logistic regression model to calculate propensity scores for receipt of treatment. 

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing treatment assignment and variables of interest for calculating weights.
        treatment_col : str
            Name of the binary treatment column (0 = control, 1 = treated). Must be of integer type. 
        cat_var : list of str, optional
            Categorical variables to be one-hot encoded. Must be of dtype 'category' and contain no missing values.
        cont_var : list of str, optional 
            Continuous variables to be imputed (median) and scaled. Must be numeric (int or float).
        binary_var : list of str, optional 
            Binary variables to be passed through without transformation. These must contain no missing values and should 
            have only two unique values (e.g., 0/1 or True/False).
        stabilized : bool, default = False
            If True, enables stabilized weights in the transform step.
        lr_kwargs : dict, optional
            Additional keyword arguments passed to sklearn's LogisticRegression.
        clip_bounds : tuple of float or list of float, optional
            If provided, clip propensity scores to this (min, max) range. 
            Common choice is (0.01, 0.99) to reduce the influence of extreme values.

        Returns
        -------
        None
            Updates internal state with propensity scores. Use `.transform()` to calculate weights

        Notes
        -----
        - This method only estimates propensity scores. Weights are calculated in `.transform()`.
        - At least one of cat_var, cont_var, or binary_var must be provided.
        """

        # Input validation 
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame.")
        
        if treatment_col not in df.columns:
            raise ValueError('treatment_col not found in df.')
        if df[treatment_col].isnull().any():
            raise ValueError('treatment_col has missing values.')
        if not set(df[treatment_col].unique()).issubset({0, 1}):
            raise ValueError('treatment_col must contain only binary values (0 and 1).')
        if not pd.api.types.is_integer_dtype(df[col]):
            raise ValueError(f"Column '{col}' must be of integer type.")
        
        if all(var is None for var in [cat_var, cont_var, binary_var]):
            raise ValueError('at least one of cat_var, cont_var, or binary_var must be provided')
        
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

        if binary_var is not None:
            # Check that columns in binary_var are present in the df
            missing = [col for col in binary_var if col not in df.columns]
            if missing:
                raise ValueError(f"The following columns in binary_var are missing from the DataFrame: {missing}")
            
            # Check that columns in binary_var have no missing values
            pt_missing = [col for col in binary_var if df[col].isnull().any()]
            if pt_missing:
                raise ValueError(f"The following columns in binary_var have missing values: {pt_missing}")
            
            # Check that all binary_var are binary (only 2 unique values)
            not_binary = [
                col for col in binary_var
                if df[col].nunique() > 2
            ]
            if not_binary:
                raise ValueError(f"The following columns in binary_var are not binary: {not_binary}")
            
            # Convert True/False to 1/0 for consistency
            for col in binary_var:
                if df[col].dtype == 'bool':
                    df[col] = df[col].astype(int)
        
        if not isinstance(stabilized, bool):
            raise ValueError("stabilized must be a boolean (True or False).")

        if clip_bounds is not None:
            if (not isinstance(clip_bounds, (tuple, list)) or
                len(clip_bounds) != 2):
                raise ValueError("clip_bounds must be a tuple or list of two float values (min, max).")

            lower, upper = clip_bounds

            if not (isinstance(lower, (int, float)) and isinstance(upper, (int, float))):
                raise ValueError("Both values in clip_bounds must be numeric.")

            if not (0 < lower < upper < 1):
                raise ValueError("clip_bounds values must be between 0 and 1 and satisfy 0 < lower < upper.")

        # Save config
        self.cat_var = cat_var or []
        self.cont_var = cont_var or []
        self.binary_var = binary_var or []
        self.all_var = self.cat_var + self.cont_var + self.binary_var

        self.treatment_col = treatment_col
        self.stabilized = stabilized
        self.clip_bounds = clip_bounds
        
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
                ('pass', 'passthrough', binary_var)],
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
        
        # Apply clipping if requested
        if self.clip_bounds is not None:
            lower, upper = self.clip_bounds
            propensity_score = np.clip(propensity_score, lower, upper)
        
        df['propensity_score'] = propensity_score
        self.propensity_score_df = df
    
    def transform(self) -> pd.DataFrame:
        """
        Calculate inverse probability of treatment weights (IPTW) based on fitted propensity scores.

        Returns
        -------
        pd.DataFrame
        A copy of the original DataFrame with the following columns:
            'propensity_score' : float
                calculated propensity scores 
            'iptw' : float
                calculated IPTW 

        Notes
        -----
        - For treated patients: weight = 1 / propensity score
        - For control patients: weight = 1 / (1 - propensity score)
        - If stabilized = True, weights are multiplied by the marginal probability of treatment or control.
        - Must call `.fit()` before calling `.transform()`.
        """
        if self.propensity_score_df is None:
            raise ValueError("Model not fitted. Please run `.fit()` first.")

        df = self.propensity_score_df.copy()

        if self.stabilized:
            p_treated = df[self.treatment_col].mean()
            df['iptw'] = np.where(
                df[self.treatment_col] == 1,
                p_treated / df['propensity_score'],
                (1 - p_treated) / (1 - df['propensity_score'])
            )

        else:
            df['iptw'] = np.where(
                df[self.treatment_col] == 1,
                1 / df['propensity_score'],
                1 / (1 - df['propensity_score'])
            )

        self.weight_df = df
        self.weight_col = "iptw"
        return df
    
    def fit_transform(self, 
                      *args, 
                      **kwargs) -> pd.DataFrame:
        """
        Fit the propensity score model and compute IPTW weights in one step.

        Returns
        -------
        pd.DataFrame
            A DataFrame with 'propensity_score' and 'iptw' columns added.

        Notes
        -----
        This is a convenience method equivalent to calling `.fit()` followed by `.transform()`.
        """
        self.fit(*args, **kwargs)
        return self.transform()
    
    def ps_plot(self,
                bins: int = 20): 
        """
        Generates a propensity score overlap plot for treatment vs control. 

        Parameters
        ----------
        bins : int, default = 20
            Number of bins for the histogram

        This method uses internal attributes set during the .fit() or fit_transform() calls:
            - self.propensity_score_df : the DataFrame with variables, treatment, and propensity scores 

        Returns
        -------
        matplotlib.figure.Figure 
            A histogram plot showing raw propensity scores by treatment group. 
        """
        if self.propensity_score_df is None:
            raise ValueError("propensity_score_df is None. Please call `.fit()` or `.fit_transform()` first.")
        
        df = self.propensity_score_df
        treatment_col = self.treatment_col

        fig, ax = plt.subplots(figsize=(8, 5))

        # Histogram for treated patients
        ax.hist(df[df[treatment_col] == 1]['propensity_score'], 
                 bins = bins, 
                 alpha = 0.3, 
                 label = 'Treatment', 
                 color = 'blue',
                 edgecolor='black')
        
        # Histogram for untreated patients (horizontal, with negative counts to "flip" it)
        ax.hist(df[df[treatment_col] == 0]['propensity_score'], 
                 bins = bins, 
                 weights= -np.ones_like(df[df[treatment_col] == 0]['propensity_score']),
                 alpha = 0.3, 
                 label = 'Control', 
                 color = 'green', 
                 edgecolor = 'black')

        # Adding titles and labels
        ax.set_title('Propensity Score Distribution by Treatment Group', pad = 25, size = 18, weight = 'bold')
        ax.set_xlabel('Propensity Score', labelpad = 15, size = 12, weight = 'bold')
        ax.set_ylabel('Count', labelpad = 15, size = 12, weight = 'bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        yticks = ax.get_yticks()
        ax.set_yticklabels([f'{abs(int(tick))}' for tick in yticks])

        ax.legend(prop = {'size': 10})

        fig.tight_layout()
        return fig
        
    def smd(self,
            return_fig: bool = False):
        """
        Compute and plots standardized mean differences (SMDs) before and after weighting for all variables used in the IPTW model.

        Parameters
        ----------
        return_fig : bool, default = False
            If True, returns a Love plot of the SMDs, which is a dot plot with variable names on the y-axis and standardized mean 
            differences on the x-axis.

        This method uses internal attributes set during the .fit() and .transform() or fit_transform() calls:
            - self.weights_df : the DataFrame with variables, treatment, and weights
            - self.cat_var : list of categorical variables
            - self.cont_var : list of continuous variables
            - self.binary_var : list of binary variables

        Returns
        -------
        pd.DataFrame
            Returned only if `return_df=True`. Contains:
                - variable : variable name
                - smd_unweighted : SMD with no weights
                - smd_weighted : SMD using IPTW 

        matplotlib.figure.Figure (optional)
            Returned only if `return_fig=True`.
            A plot showing unweighted and weighted SMDs for all included variables.

        Notes
        -----
        SMDs quantify the difference in variable distributions between treated and control groups.

        For continuous variables:
            SMD = (mean_treated - mean_control) / sqrt[(sd_treated² + sd_control²)/2]

        For categorical and binary variables:
            SMD = (p_treated - p_control) / sqrt[(p_treated * (1 - p_treated) + p_control * (1 - p_control)) / 2]

        Where:
            - mean_treated / mean_control = means of the variable in each group
            - sd_treated / sd_control = standard deviations
            - p_treated / p_control = proportion of group members in a given category or with value == 1
        
        Median is imputed for missing continuous variables. 
        Categorical variables are one-hot-encoded.
        """
        # Input validation 
        if not self.all_var:
            raise ValueError("No variables found. Please run .fit() or fit_transform()")

        if self.weight_df is None:
            raise ValueError("No weights found. Please run fit_transform()")
        
        if not isinstance(return_fig, bool):
            raise ValueError("return_fig must be a boolean (True or False).")

        smd_df = self.weight_df[self.all_var + ['treatment', 'itpw']].copy()

        # Calculate SMD for continuous variables 
        smd_cont = []
        for var in self.cont_var:
            # Imput median for missing 
            smd_df[var] = smd_df[var].fillna(smd_df[var].median())

            treat_mask = smd_df['treatment'] == 1
            control_mask = smd_df['treatment'] == 0

            # Unweighted
            m1 = smd_df.loc[treat_mask, var].mean()
            m0 = smd_df.loc[control_mask, var].mean()
            s1 = smd_df.loc[treat_mask, var].std()
            s0 = smd_df.loc[control_mask, var].std()

            pooled_sd = np.sqrt(0.5 * (s1**2 + s0**2))
            smd_unweighted = (m1 - m0) / pooled_sd if pooled_sd > 0 else 0.0

            # Weighted
            m1_w = np.average(smd_df.loc[treat_mask, var], weights=smd_df.loc[treat_mask, 'iptw'])
            m0_w = np.average(smd_df.loc[control_mask, var], weights=smd_df.loc[control_mask, 'iptw'])
            s1_w = np.sqrt(np.average((smd_df.loc[treat_mask, var] - m1_w) ** 2, weights=smd_df.loc[treat_mask, 'iptw']))
            s0_w = np.sqrt(np.average((smd_df.loc[control_mask, var] - m0_w) ** 2, weights=smd_df.loc[control_mask, 'iptw']))

            pooled_sd_w = np.sqrt(0.5 * (s1_w**2 + s0_w**2))
            smd_weighted = (m1_w - m0_w) / pooled_sd_w if pooled_sd_w > 0 else 0.0

            smd_cont.append({
                'variable': var,
                'smd_unweighted': smd_unweighted,
                'smd_weighted': smd_weighted
            })

        # Calculate SMD for categorical variables 
        smd_cat = []
        for var in self.cat_var: 
            # One-hot encode categories
            categories = smd_df[var].dropna().unique()
            for cat in categories:
                var_cat = f"{var}__{cat}"
                treat_mask = (smd_df['treatment'] == 1)
                control_mask = (smd_df['treatment'] == 0)
                smd_df[var_cat] = (smd_df[var] == cat).astype(int)

                # Unweighted
                p1 = smd_df.loc[treat_mask, var_cat].mean()
                p0 = smd_df.loc[control_mask, var_cat].mean()
                denom = np.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)
                smd_unweighted = (p1 - p0) / denom if denom > 0 else 0.0

                # Weighted
                p1_w = np.average(smd_df.loc[treat_mask, var_cat], weights=smd_df.loc[treat_mask, 'iptw'])
                p0_w = np.average(smd_df.loc[control_mask, var_cat], weights=smd_df.loc[control_mask, 'iptw'])
                denom_w = np.sqrt((p1_w * (1 - p1_w) + p0_w * (1 - p0_w)) / 2)
                smd_weighted = (p1_w - p0_w) / denom_w if denom_w > 0 else 0.0

                smd_cat.append({
                    'variable': var_cat,
                    'smd_unweighted': smd_unweighted,
                    'smd_weighted': smd_weighted
                })
        
        # Calculate SMD for binary variables 
        smd_bin = []
        for var in self.binary_var:
            treat_mask = smd_df['treatment'] == 1
            control_mask = smd_df['treatment'] == 0

            # Unweighted
            p1 = smd_df.loc[treat_mask, var].mean()
            p0 = smd_df.loc[control_mask, var].mean()
            denom = np.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)
            smd_unweighted = (p1 - p0) / denom if denom > 0 else 0.0

            # Weighted
            p1_w = np.average(smd_df.loc[treat_mask, var], weights=smd_df.loc[treat_mask, 'iptw'])
            p0_w = np.average(smd_df.loc[control_mask, var], weights=smd_df.loc[control_mask, 'iptw'])
            denom_w = np.sqrt((p1_w * (1 - p1_w) + p0_w * (1 - p0_w)) / 2)
            smd_weighted = (p1_w - p0_w) / denom_w if denom_w > 0 else 0.0

            smd_bin.append({
                'variable': var,
                'smd_unweighted': smd_unweighted,
                'smd_weighted': smd_weighted
            })

        smd_results_df = pd.DataFrame(smd_cont + smd_cat + smd_bin)
        smd_results_df['smd_unweighted'] = smd_results_df['smd_unweighted'].abs()
        smd_results_df['smd_weighted'] = smd_results_df['smd_weighted'].abs()
        smd_results_df = smd_results_df.sort_values(by = 'smd_unweighted', ascending = True).reset_index(drop = True)

        self.smd_results_df = smd_results_df
        
        if return_fig:
            fig, ax = plt.subplots(figsize=(8, 0.4 * len(smd_results_df) + 2))
            
            # Plot points
            ax.scatter(smd_results_df['smd_unweighted'], smd_results_df['variable'], label = 'Unweighted', color = 'red')
            ax.scatter(smd_results_df['smd_weighted'], smd_results_df['variable'], label = 'Weighted', color = 'skyblue')

            # Reference lines
            ax.axvline(x = 0, color = 'black', linestyle = '-', linewidth = 2, alpha = 0.5) 
            ax.axvline(x = 0.1, color = 'black', linestyle = '--', linewidth = 2, alpha = 0.5) 

            # Axis labels and limits
            ax.set_xlabel('Absolute Standardized Mean Difference', labelpad = 15, size = 12, weight = 'bold')
            ax.set_xlim(-0.02)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Title legend
            ax.set_title('Love Plot: Variable Balance', pad = 20, size = 18, weight = 'bold')
            ax.legend(prop = {'size': 10})

            fig.tight_layout()
            
            return smd_results_df, fig
        
        else:
            return smd_results_df
        
    def survival_metrics(self,
                         df: pd.DataFrame,
                         weight_col: Optional[str] = None, 
                         duration_col: str,
                         event_col: str,
                         psurv_time_points: Optional[List[float]] = None, 
                         rmst_time_points: Optional[List[float]] = None,
                         median_time: Optional[bool] = False,
                         n_bootstrap: int = 1000,
                         random_state: Optional[int] = None) -> dict:

        """
        Estimate survival metrics for treatment group, control group, and their difference using 
        IPTW-adjusted Kaplan-Meier analysis with bootstrapped 95% confidence intervals.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing survival data, treatment assignment, and all variables used 
            in the IPTW model. This should match the structure of the dataframe used in the original 
            fit()/fit_transform() call.
        weight_col : str, optional 
             Name of the column in `df` containing the IPTW weights. If not provided, defaults to the 
            column name used in `.fit_transform()` (typically "iptw"). 
            If you have renamed the weight column or are using a modified dataframe, you must specify 
            the correct column name explicitly. The column must contain non-negative numeric weights.
        duration_col : str
            Name of the column containing the survival duration (e.g., time to event or censoring).
        event_col : str
            Name of the column indicating event occurrence (1 = event, 0 = censored). Must be of integer type. 
        psurv_time_points : list of float, optional
            Specific time points (in same units as duration_col) at which to estimate survival probability.
            If None, no survival probabilities will be calculated.
        rmst_time_points : list of float, optional
            Specific time horizons (in same units as duration_col) for restricted mean survival time (RMST) estimation.
            If None, RMST will not be calculated.
        median_time : bool, optional
            Whether to calculate median survival time (e.g., median overall survival) with bootstrapped confidence intervals.
        n_bootstrap : int, default=1000
            Number of bootstrap iterations used to estimate confidence intervals.
        random_state : int, optional
            Seed for reproducibility of bootstrap resampling.

        Returns
        -------
        results : dict
            Dictionary containing survival metrics and 95% confidence intervals for treatment and control groups,
            as well as the difference between them. Example format:
            {
                'treatment': {
                    'median': (est, lower_ci, upper_ci),
                    'survival_prob': {6: (est, lci, uci), 12: (est, lci, uci)},
                    'rmst': {24: (est, lci, uci)}
                },
                'control': {
                    ...
                },
                'difference': {
                    ...
                }
            }

        Notes
        -----
        This method requires that .fit() or .fit_transform() has been run prior to use. During each bootstrap iteration, weights 
        are recalculated using the variables provided in the original .fit() or .fit_transform() call, including stabilization and 
        propensity score clipping preferences.
        """

        # Input validation for df 
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        if self.weight_df is None:
            raise ValueError("No model state found. Please run .fit() or .fit_transform() before calling survival_metrics().")
        if df.shape[0] != self.weight_df.shape[0]:
            raise ValueError("df appears to be a different size than the one submitted in .fit() or .fit_transform().")
        missing_vars = [col for col in self.all_var if col not in df.columns]
        if missing_vars:
            raise ValueError(f"The following variables used in the IPTW model are missing from df: {missing_vars}.")

        # Input validation for weight_col and duration_col
        if weight_col is None:
            weight_col = self.weight_col
        for col in [weight_col, duration_col]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe.")
            if df[col].isnull().any():
                raise ValueError(f"Column '{col}'contains missing values.")
            if (df[col] < 0).any():
                raise ValueError(f"Column '{col}' must contain non-negative values only.")
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column '{col}' must be numeric (int or float).")
            
        # Input validation for treatment_col and event_col
        for col in [self.treatment_col, event_col]: 
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe.")
            if df[col].isnull().any():
                raise ValueError(f"Column '{col}' has missing values.")
            if not set(df[col].unique()).issubset({0, 1}):
                raise ValueError(f"Column '{col}' must contain only binary values (0 and 1).")
            if not pd.api.types.is_integer_dtype(df[col]):
                raise ValueError(f"Column '{col}' must be of integer type.")

        if (
            psurv_time_points is None 
            and rmst_time_points is None 
            and not median_time
        ):
            raise ValueError("At least one of psurv_time_points, rmst_time_points, or median_time must be specified.")

        # Input validation for psurv_time_points and rmst_time_points
        for name, arg in [("psurv_time_points", psurv_time_points), 
                          ("rmst_time_points", rmst_time_points)]:
            if arg is not None:
                if not isinstance(arg, list):
                    raise ValueError(f"{name} must be a list of floats or ints.")
                if not all(isinstance(t, (float, int)) for t in arg):
                    raise ValueError(f"All values in {name} must be numeric (float or int).")
                if any(t <= 0 for t in arg):
                    raise ValueError(f"All values in {name} must be positive.")

        # Input validation for median_time
        if not isinstance(median_time, bool):
            raise ValueError("median_time must be a boolean (True or False).")

        # Input validation for n_bootstrap
        if not isinstance(n_bootstrap, int) or n_bootstrap <= 0:
            raise ValueError("n_bootstrap must be a positive integer.")

        # Input validation for random_state
        if random_state is not None:
            if not isinstance(random_state, int):
                raise ValueError("random_state must be an integer or None.")

            


        