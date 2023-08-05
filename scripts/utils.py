import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from sklearn.model_selection import cross_validate
import numpy as np
import joblib

import torch

def get_column_info(df):
    """
    This function provides detailed information about each column in a given pandas DataFrame. 
    For each column, it calculates the number of unique values, the number of null values, and determines the data type.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame for which to gather column information.

    Returns:
    pandas.DataFrame: A DataFrame where each row corresponds to a column in the input DataFrame.
    The columns of the returned DataFrame are 'Column' (column name), 'Unique Values' (number of unique values), 
    'Null Values' (number of null or NaN values), and 'Type' (data type of the column).
    The returned DataFrame is sorted first by the number of unique values, and then by the number of null values.

    Prints:
    Tuple: The shape of the input DataFrame (number of rows, number of columns).
    Tuple: The shape of the returned DataFrame (should be equal to the number of columns in the input DataFrame, 4).
    """
    column_info = []
    for col in df.columns:
        unique_values = df[col].nunique()
        null_values = df[col].isnull().sum()
        value_type = df[col].dtype
        column_info.append({'Column': col, 'Unique Values': unique_values, 'Null Values': null_values, 'Type': value_type})

    column_info_df = pd.DataFrame(column_info).sort_values('Unique Values')
    column_info_df = column_info_df.sort_values("Null Values")
    print(df.shape)
    print(column_info_df.shape)
    return column_info_df


def assert_disjoint_sets(set_list):
    """
    This function checks whether all sets in a given list of sets are disjoint from each other.
    If two sets are not disjoint, it raises an AssertionError.

    Parameters:
    set_list (list of set): The list of sets to check for disjointness. Each element of the list should be a set.

    Returns:
    None.

    Raises:
    AssertionError: If any two sets in the list are not disjoint. The error message includes the two sets that are not disjoint.
    """
    for i in range(len(set_list)):
        for j in range(i+1, len(set_list)):
            assert set_list[i].isdisjoint(set_list[j]), f"Sets {set_list[i]} and {set_list[j]} are not disjoint."

def subtract_subsets_from_superset(superset, subset_list):
    """
    This function subtracts a list of subsets from a given superset, and returns the remainder set.

    Parameters:
    superset (set): The superset from which to subtract the subsets.
    subset_list (list of set): The list of subsets to subtract from the superset. Each element of the list should be a set.

    Returns:
    set: The remainder set after subtracting all subsets from the superset.
    """
    remainder_set = superset.copy()  # make a copy of the superset
    
    for subset in subset_list:
        remainder_set -= subset  # subtract each subset from the remainder set
        
    return remainder_set


def mape_score(true, pred):
    """
    Computes the Mean Absolute Percentage Error (MAPE) between true and predicted values.

    Parameters
    ----------
    true : numpy array or pandas Series
        The array of true values.

    pred : numpy array or pandas Series
        The array of predicted values. The length of `pred` must be equal to the length of `true`.

    Returns
    -------
    float
        The Mean Absolute Percentage Error (MAPE) between the true and predicted values.

    Notes
    -----
    The Mean Absolute Percentage Error (MAPE) is a scale-independent measure of error, 
    which makes it a useful metric for comparing the relative error across different series. 
    However, it can lead to division by zero if a true value is zero. 
    """
    return np.mean(np.abs((pred - true) / true))


def mean_std_cross_val_scores(model, X_train, y_train, model_scoring, std_dev = False, **kwargs):
    """
    Performs cross-validation on a given model and training data, then calculates and returns the mean and standard deviation of the cross-validation scores.

    Parameters
    ----------
    model : scikit-learn estimator object
        The machine learning model to evaluate. This should be an instance of a scikit-learn model that implements the 'fit' method.

    X_train : numpy array or pandas DataFrame
        The feature matrix for the training data. Rows correspond to samples and columns correspond to features.

    y_train : numpy array or pandas Series
        The target values for the training data. Each value corresponds to the target value for a sample in X_train.

    model_scoring : string, callable, list/tuple or dict
        Strategy to evaluate the performance of the cross-validated model on the test set. If model_scoring is a single string or a callable to evaluate the predictions on the test set, the cross_validate function will only return a single score. If model_scoring is a list/tuple of strings or a dict of scorer names mapped to scorer callables, the cross_validate function will return a dict of scores.

    **kwargs : dict, optional
        Additional parameters to pass to the cross_validate function.

    Returns
    -------
    pandas Series
        A pandas Series where the index is the names of the scoring metrics and the values are strings representing the mean score and the standard deviation in the format: "mean (+/- std)".
    """
    scores = cross_validate(
        model, 
        X_train, 
        y_train, 
        scoring = model_scoring,
        **kwargs
    )
    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []

    if std_dev:
        for i in range(len(mean_scores)):
            out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores[i], std_scores[i])))
    else:
        for i in range(len(mean_scores)):
            out_col.append((f"%0.3f" % (mean_scores[i])))

    return pd.Series(data=out_col, index=mean_scores.index)


def evaluate_model(model, X_test, y_test):
    """
    Evaluates an already fitted model on the test data.

    Parameters
    ----------
    model : scikit-learn estimator object
        The fitted machine learning model to evaluate. This should be an instance of a scikit-learn model that implements the 'predict' method.

    X_test : numpy array or pandas DataFrame
        The feature matrix for the test data. Rows correspond to samples and columns correspond to features.

    y_test : numpy array or pandas Series
        The target values for the test data. Each value corresponds to the target value for a sample in X_test.

    Returns
    -------
    dict
        A dictionary where the keys are the names of the scoring metrics and the values are the scores.
    """
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mape = mape_score(y_test, predictions)
    return {"eval_r2_score": r2, "eval_mape": mape}


from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np

def impute_data_with_knn(df, categorical_cols, ordinal_cols, numerical_cols, binary_cols, n_neighbors=3):
    """
    Function to impute missing values. For categorical and ordinal features, fill with the mode.
    For numerical and binary features, use KNN imputation.

    Parameters:
    df (pandas.DataFrame): The input DataFrame with missing values.
    categorical_cols (list): List of categorical column names.
    ordinal_cols (list): List of ordinal column names.
    numerical_cols (list): List of numerical column names.
    binary_cols (list): List of binary column names.
    n_neighbors (int): Number of neighbors to use for KNN imputation.

    Returns:
    df_imputed (pandas.DataFrame): DataFrame where missing values have been imputed.
    """

    # Initialize KNN Imputer
    imputer = KNNImputer(n_neighbors=n_neighbors)

    # Fill missing values in categorical and ordinal columns with the mode
    for col in categorical_cols + ordinal_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Apply KNN imputation to numerical and binary columns
    for col in numerical_cols + binary_cols:
        df[[col]] = imputer.fit_transform(df[[col]])

    return df

# Let's train the model
def mape(y_true, y_pred): 
    return torch.mean(torch.abs((y_true - y_pred) / y_true))


# Adding Adjusted R2 scores for the model since we have very high number of features
def adj_r2_score(y_true, y_pred, n, p):
    r2 = np.corrcoef(y_true, y_pred)[0, 1]**2
    adj_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
    return adj_r2
        

def save_model(model, filename):
    """
    Save the model to a file.
    """
    joblib.dump(model, filename)

def load_model(filename):
    """
    Load the model from a file.
    """
    return joblib.load(filename)
