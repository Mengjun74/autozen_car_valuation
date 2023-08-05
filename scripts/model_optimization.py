from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from base_step import VERBOSE
import os
import warnings
import pandas as pd
from utils import mape_score
from utils import save_model, load_model
from feature_engineering import FeatureEngineer
from model_ml_training import ModelTrainerML
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from xgboost.sklearn import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


class ModelTuner(FeatureEngineer):
    def __init__(self, data_dir, random_state=591, test_size=0.2,num_of_steps=6):
        super().__init__(data_dir=data_dir, 
                         db_creds=None, 
                         num_of_steps=num_of_steps, 
                         progress_message='ModelTuner: processing...', 
                         random_state=random_state,
                         test_size=test_size)

    def tuneXGBoost(self, reuse_optimized_params = True):
        model_trainer = ModelTrainerML(data_dir=self.data_dir,num_of_steps=7) 
        model_trainer_results, lasso_dict = model_trainer.trainByName(['lasso_xgb_regressor'])
        if reuse_optimized_params:
            n_neighbours = 17
            learning_rate = 0.08
            alpha = 0.01
            n_estimators = 1000
        else:
            param_grid = {
                'pipeline__knnimputer__n_neighbors': [5,7,9,10,11,12,17,18],  # Searching among 5 to 20 neighbors for imputation
                'pipeline__xgbregressor__alpha': [1e-3, 1e-2, 1e-1, 1, 10, 100],  # Regularization for Lasso
                'pipeline__xgbregressor__n_estimators': list(range(100, 1001, 100)),  # Number of gradient boosted trees. Equivalent to number of boosting rounds
                'pipeline__xgbregressor__learning_rate': [x / 100.0 for x in range(1, 61)],  # Boosting learning rate (xgb’s “eta”)
            }
            best_params=None
            best_model, best_params = model_trainer.tunePipe('lasso_xgb_regressor',param_grid,n_iter=5)
            n_neighbours = best_params['pipeline__knnimputer__n_neighbors']
            learning_rate = best_params['pipeline__xgbregressor__learning_rate']
            alpha = best_params['pipeline__xgbregressor__alpha']
            n_estimators = best_params['pipeline__xgbregressor__n_estimators']
        optimized_model = {
            "lasso_xgb_regressor_opt": make_pipeline(
                    StandardScaler(),
                    KNNImputer(n_neighbors=n_neighbours),
                    StandardScaler(),
                    SelectFromModel(Lasso(alpha=5, max_iter=1000)),
                    XGBRegressor(objective="reg:linear", 
                                verbosity=0, 
                                learning_rate = learning_rate,
                                alpha = alpha,
                                n_estimators = n_estimators,
                                random_state=591),
            )
        }
        model_trainer_results_opt, lasso_dict = model_trainer.trainByDict(optimized_model,feature_importance=True)
        # Comparing before and after
        df = pd.concat([model_trainer_results.T, model_trainer_results_opt.T], axis=1)
        print(f"\nOptimal Hyper-params:")
        print(f"  n_neighbours={n_neighbours}")
        print(f"  learning_rate={learning_rate}")
        print(f"  alpha={alpha}")
        print(f"  n_estimators={n_estimators}")
        # saving model
        save_model(optimized_model,os.path.join(os.getcwd(),".","models","xgb_tuned_model"))
        return optimized_model, df
    
    def loadXGBoostBestModel(self):
        saved_model = load_model(os.path.join(os.getcwd(),".","models","xgb_tuned_model"))['lasso_xgb_regressor_opt']
        saved_preprocessor = load_model(os.path.join(os.getcwd(),".","models","preprocessor"))
        pipe = make_pipeline(saved_preprocessor, saved_model)
        return pipe
    
    def tuneQuantileRegressors(self,lower=0.125, upper=0.875):
        n_neighbours = 17
        learning_rate = 0.08
        n_estimators = 1000
        lower_bound_model_dict = {"lower_quantile": make_pipeline(
                StandardScaler(),
                KNNImputer(n_neighbors=n_neighbours),
                StandardScaler(),
                #IterativeImputer(random_state=0, max_iter=1,verbose=2,estimator = RandomForestRegressor()),
                SelectFromModel(Lasso(alpha=5, max_iter=1000)),
                GradientBoostingRegressor(loss="quantile", alpha=0.125,learning_rate=learning_rate,n_estimators=n_estimators),
        )}
        upper_bound_model_dict  = {"upper_quantile": make_pipeline(
                StandardScaler(),
                KNNImputer(n_neighbors=n_neighbours),
                StandardScaler(),
                #IterativeImputer(random_state=0, max_iter=1,verbose=2,estimator = RandomForestRegressor()),
                SelectFromModel(Lasso(alpha=5, max_iter=1000)),
                GradientBoostingRegressor(loss="quantile", alpha=0.875,learning_rate=learning_rate,n_estimators=n_estimators),
        )}
        model_trainer = ModelTrainerML(data_dir=self.data_dir,num_of_steps=7) 
        model_trainer.trainByDict(lower_bound_model_dict);
        model_trainer.trainByDict(upper_bound_model_dict);
        save_model(lower_bound_model_dict,os.path.join(os.getcwd(),".","models","xgb_tuned_quantile_lower"))
        save_model(upper_bound_model_dict,os.path.join(os.getcwd(),".","models","xgb_tuned_quantile_upper"))
    
    def predictXGBoost(self,pipe,sample):
        return pipe.predict(sample)[0]
    
    def loadXGBoostIntervalPredictors(self):
        lower_saved_model = load_model(os.path.join(os.getcwd(),".","models","xgb_tuned_quantile_lower"))['lower_quantile']
        upper_saved_model = load_model(os.path.join(os.getcwd(),".","models","xgb_tuned_quantile_upper"))['upper_quantile']
        saved_preprocessor = load_model(os.path.join(os.getcwd(),".","models","preprocessor"))
        lower_bound_pipeline = make_pipeline(saved_preprocessor, 
                                    lower_saved_model)
        upper_bound_pipeline = make_pipeline(saved_preprocessor, 
                                             upper_saved_model)
        return lower_bound_pipeline, upper_bound_pipeline
    
    def predictXGBoostInterval(self,lower_bound_pipeline,upper_bound_pipeline,sample):
        lower_bound = lower_bound_pipeline.predict(sample)[0]
        upper_bound = upper_bound_pipeline.predict(sample)[0]
        return lower_bound,upper_bound
    
if __name__ == '__main__':
    notebook_dir = os.getcwd()
    data_dir = os.path.join(notebook_dir, ".", "data")
    model_tuner= ModelTuner(data_dir)
    best_model, results_df = model_tuner.tuneXGBoost()
    model_tuner.tuneQuantileRegressors()
    print(results_df)
    
    

