# feature_engineering.py
from base_step import PipelineStep
from base_step import VERBOSE
import os
import pandas as pd
import numpy as np
from progress.bar import ChargingBar
from autozen_features import AutozenFeatures
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings("ignore")

class FeatureEngineer (PipelineStep):
    """
    This class is designed to perform feature engineering on the dataset.

    It extends the PipelineStep class and is responsible for preprocessing data, handling missing values,
    and splitting the dataset into training and testing sets.
    """
    def __init__(self, data_dir, db_creds=None, num_of_steps=1, progress_message="Progress msg here ...",
                 random_state=591, test_size=0.2):
        super().__init__(data_dir,
                         db_creds=db_creds,
                         num_of_steps=num_of_steps,
                         progress_message=progress_message)
        
        # Define input and output
        self.MERGED_PROCESSED_WON = os.path.join(self.data_dir, "processed_az_auctioned_won.csv")
        self.random_state = random_state
        self.test_size = test_size
        self.df_won = pd.read_csv(self.MERGED_PROCESSED_WON)
        self.preprocessor = None
        self.df_won_imputed = None

        # Define data containers
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    
    @staticmethod
    def _impute(df, cols, imputer):
        """
        Generic function to impute missing values in the given DataFrame using the given Imputer.
        """
        for col in cols['categorical_cols'] + cols['ordinal_cols']:
            df[col].fillna(df[col].mode()[0], inplace=True)
        for col in cols['numerical_cols'] + cols['binary_cols']:
            df[[col]] = imputer.fit_transform(df[[col]])
        return df
    
    @staticmethod
    def impute_using_knn(df, n_neighbors=5):
        """
        Function to impute missing values in categorical, ordinal, numerical, and binary columns using KNN Imputer.
        """
        imputer = KNNImputer(n_neighbors=n_neighbors)
        return FeatureEngineer._impute(df, FeatureEngineer.cols, imputer)

    @staticmethod
    def impute_using_iterative_imputer(df):
        """
        Function to impute missing values in categorical, ordinal, numerical, and binary columns using Iterative Imputer.
        """
        imputer = IterativeImputer()
        return FeatureEngineer._impute(df, FeatureEngineer.cols, imputer)
    
    @staticmethod 
    def impute_no_impute(df):
        return df
    
    def _get_preprocessor(self):
        """
        Function to get or create the column transformer for preprocessing the data.
        """        
        if self.preprocessor == None:
            self.preprocessor = make_column_transformer(
            #(transformer,list of features)
                (AutozenFeatures.numeric_feat_transformer, AutozenFeatures.numeric_feat_list),
                (AutozenFeatures.ordinal_feat_transformer, AutozenFeatures.ordinal_feat_list),
                (AutozenFeatures.categorical_feat_transformer, AutozenFeatures.categorical_feat_list),
                (AutozenFeatures.ordinal_feat_a_transformer, AutozenFeatures.ordinal_feat_list_a),
                (AutozenFeatures.ordinal_feat_b_transformer, AutozenFeatures.ordinal_feat_list_b),
                (AutozenFeatures.binary_feat_transformer, AutozenFeatures.binary_feat_list),
                ("drop", ["created_at"])
            )
        return self.preprocessor

    def _train_test_split(self):
        """
        Function to split the dataset into training and testing sets.
        """
        self.X = self.df_won_imputed.drop(columns=['bid_winning'])
        self.y = self.df_won_imputed['bid_winning']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state
        )

    def _impute_missing(self, method='none'):
        """
        Function to impute missing values using the specified method. If method is not specified, it does not perform any imputation.
        """
        # Drop irrelevant features
        self.df_won_imputed = self.df_won.drop(AutozenFeatures.to_drop_feat_list, axis=1)
        
        if 'none' == method:
            self.df_won_imputed = self.impute_no_impute(self.df_won_imputed)
        elif 'knn:mnar' == method:
            self.df_won_imputed = self.impute_using_knn(self.df_won_imputed)
            self.progressNext(f'ModelTrainer: imputed missing data using knn:mnar')
        elif 'iterative:mnar' == method:
            self.df_won_imputed = self.impute_using_iterative_imputer(self.df_won_imputed)
            self.progressNext(f'ModelTrainer: imputed missing data using iterative:mnar')
        else:
            raise Exception(f"'{method}' imputation method not supported'")
        
    # all columns
    cols = {'categorical_cols':AutozenFeatures.categorical_feat_list,
            'ordinal_cols':AutozenFeatures.ordinal_feat_list_a + AutozenFeatures.ordinal_feat_list_b + AutozenFeatures.ordinal_feat_list + AutozenFeatures.binary_feat_list,
            'numerical_cols':AutozenFeatures.numeric_feat_list,
            'binary_cols':[]}
# feature_engineering.py
from base_step import PipelineStep
from base_step import VERBOSE
import os
import pandas as pd
import numpy as np
from progress.bar import ChargingBar
from autozen_features import AutozenFeatures
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings("ignore")

class FeatureEngineer (PipelineStep):
    """
    This class is designed to perform feature engineering on the dataset.

    It extends the PipelineStep class and is responsible for preprocessing data, handling missing values,
    and splitting the dataset into training and testing sets.
    """
    def __init__(self, data_dir, db_creds=None, num_of_steps=1, progress_message="Progress msg here ...",
                 random_state=591, test_size=0.2):
        super().__init__(data_dir,
                         db_creds=db_creds,
                         num_of_steps=num_of_steps,
                         progress_message=progress_message)
        
        # Define input and output
        self.MERGED_PROCESSED_WON = os.path.join(self.data_dir, "processed_az_auctioned_won.csv")
        self.random_state = random_state
        self.test_size = test_size
        self.df_won = pd.read_csv(self.MERGED_PROCESSED_WON)
        self.preprocessor = None
        self.df_won_imputed = None

        # Define data containers
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    
    @staticmethod
    def _impute(df, cols, imputer):
        """
        Generic function to impute missing values in the given DataFrame using the given Imputer.
        """
        for col in cols['categorical_cols'] + cols['ordinal_cols']:
            df[col].fillna(df[col].mode()[0], inplace=True)
        for col in cols['numerical_cols'] + cols['binary_cols']:
            df[[col]] = imputer.fit_transform(df[[col]])
        return df
    
    @staticmethod
    def impute_using_knn(df, n_neighbors=5):
        """
        Function to impute missing values in categorical, ordinal, numerical, and binary columns using KNN Imputer.
        """
        imputer = KNNImputer(n_neighbors=n_neighbors)
        return FeatureEngineer._impute(df, FeatureEngineer.cols, imputer)

    @staticmethod
    def impute_using_iterative_imputer(df):
        """
        Function to impute missing values in categorical, ordinal, numerical, and binary columns using Iterative Imputer.
        """
        imputer = IterativeImputer()
        return FeatureEngineer._impute(df, FeatureEngineer.cols, imputer)
    
    @staticmethod 
    def impute_no_impute(df):
        return df
    
    def _get_preprocessor(self):
        """
        Function to get or create the column transformer for preprocessing the data.
        """        
        if self.preprocessor == None:
            self.preprocessor = make_column_transformer(
            #(transformer,list of features)
                (AutozenFeatures.numeric_feat_transformer, AutozenFeatures.numeric_feat_list),
                (AutozenFeatures.ordinal_feat_transformer, AutozenFeatures.ordinal_feat_list),
                (AutozenFeatures.categorical_feat_transformer, AutozenFeatures.categorical_feat_list),
                (AutozenFeatures.ordinal_feat_a_transformer, AutozenFeatures.ordinal_feat_list_a),
                (AutozenFeatures.ordinal_feat_b_transformer, AutozenFeatures.ordinal_feat_list_b),
                (AutozenFeatures.binary_feat_transformer, AutozenFeatures.binary_feat_list),
                ("drop", ["created_at"])
            )
        return self.preprocessor

    def _train_test_split(self):
        """
        Function to split the dataset into training and testing sets.
        """
        self.X = self.df_won_imputed.drop(columns=['bid_winning'])
        self.y = self.df_won_imputed['bid_winning']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state
        )

    def _impute_missing(self, method='none'):
        """
        Function to impute missing values using the specified method. If method is not specified, it does not perform any imputation.
        """
        # Drop irrelevant features
        self.df_won_imputed = self.df_won.drop(AutozenFeatures.to_drop_feat_list, axis=1)
        
        if 'none' == method:
            self.df_won_imputed = self.impute_no_impute(self.df_won_imputed)
        elif 'knn:mnar' == method:
            self.df_won_imputed = self.impute_using_knn(self.df_won_imputed)
            self.progressNext(f'ModelTrainer: imputed missing data using knn:mnar')
        elif 'iterative:mnar' == method:
            self.df_won_imputed = self.impute_using_iterative_imputer(self.df_won_imputed)
            self.progressNext(f'ModelTrainer: imputed missing data using iterative:mnar')
        else:
            raise Exception(f"'{method}' imputation method not supported'")
        
    # all columns
    cols = {'categorical_cols':AutozenFeatures.categorical_feat_list,
            'ordinal_cols':AutozenFeatures.ordinal_feat_list_a + AutozenFeatures.ordinal_feat_list_b + AutozenFeatures.ordinal_feat_list + AutozenFeatures.binary_feat_list,
            'numerical_cols':AutozenFeatures.numeric_feat_list,
            'binary_cols':[]}
