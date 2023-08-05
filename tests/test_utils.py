# python -m unittest tests.test_utils
# test_utils.py
import unittest
import pandas as pd
import numpy as np
import sys
sys.path.append('./scripts')
from utils import get_column_info, assert_disjoint_sets, subtract_subsets_from_superset, mape_score, mean_std_cross_val_scores, evaluate_model, impute_data_with_knn, mape, adj_r2_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import torch

class TestUtils(unittest.TestCase):

    def test_get_column_info(self):
        df = pd.DataFrame({'A': [1, 2, np.nan, 4], 
                           'B': ['a', 'b', 'b', 'a'], 
                           'C': [1.1, 1.1, 1.2, 1.3]})
        info_df = get_column_info(df)
        self.assertEqual(info_df.shape, (3, 4))

    def test_assert_disjoint_sets(self):
        sets = [{1, 2, 3}, {4, 5, 6}, {7, 8, 9}]
        # This should not raise an AssertionError
        assert_disjoint_sets(sets)

    def test_subtract_subsets_from_superset(self):
        superset = {1, 2, 3, 4, 5}
        subsets = [{1, 2}, {3, 4}]
        remainder = subtract_subsets_from_superset(superset, subsets)
        self.assertEqual(remainder, {5})

    def test_mape_score(self):
        true = np.array([1, 2, 3, 4, 5])
        pred = np.array([1.1, 2.2, 2.9, 3.8, 5.2])
        mape = mape_score(true, pred)
        self.assertAlmostEqual(mape, 0.0647, places=3)

    def test_mean_std_cross_val_scores(self):
        X, y = make_regression(n_samples=100, n_features=20, noise=0.1)
        model = LinearRegression()
        scores = mean_std_cross_val_scores(model, X, y, 'neg_mean_squared_error', cv=5)
        self.assertTrue('test_score' in scores.index)

    def test_evaluate_model(self):
        X, y = make_regression(n_samples=100, n_features=20, noise=0.1)
        model = LinearRegression().fit(X, y)
        scores = evaluate_model(model, X, y)
        self.assertTrue('eval_r2_score' in scores.keys())

    def test_impute_data_with_knn(self):
        df = pd.DataFrame({'A': [1, 2, np.nan, 4], 'B': [0, 1, 1, 0], 'C': [1.1, 1.1, np.nan, 1.3]})
        categorical_cols = ['A']
        ordinal_cols = []
        numerical_cols = ['C']
        binary_cols = ['B']
        df_imputed = impute_data_with_knn(df, categorical_cols, ordinal_cols, numerical_cols, binary_cols)
        self.assertFalse(df_imputed.isnull().any().any())

    def test_mape(self):
        y_true = torch.Tensor([1, 2, 3, 4, 5])
        y_pred = torch.Tensor([1.1, 2.2, 2.9, 3.8, 5.2])
        mape_val = mape(y_true, y_pred)
        self.assertAlmostEqual(mape_val.item(), 0.0646, places=3)
    
    def test_adj_r2_score(self):
        y_true = np.array([3, -0.5, 2, 7, 4.2])
        y_pred = np.array([2.5, 0.0, 2.1, 7.8, 5.3])
        n = 5  # Number of observations
        p = 1  # Number of predictors
        adj_r2 = adj_r2_score(y_true, y_pred, n, p)
        self.assertAlmostEqual(adj_r2, 0.95042, places=3)



