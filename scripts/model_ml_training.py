# model_ml_training.py
from base_step import PipelineStep
from base_step import VERBOSE
import os
import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from utils import mean_std_cross_val_scores, mape_score, evaluate_model, adj_r2_score
from sklearn.metrics import make_scorer
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from sklearn.ensemble import VotingRegressor
from feature_engineering import FeatureEngineer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from utils import save_model
warnings.filterwarnings("ignore")

class ModelTrainerML (FeatureEngineer):
    """
    Class for training multiple machine learning models and evaluating their performance.
    """
    def __init__(self, data_dir,random_state=591,test_size=0.2,num_of_steps=16):
        super().__init__(data_dir=data_dir, db_creds=None,num_of_steps=num_of_steps, 
                         progress_message='ModelTrainerML: processing...', 
                         random_state=random_state,test_size=test_size)
        self.models = None
        self.models_with_lasso = None
        self.scoring = None
        self.results_df = None
    
    def printHelp(self):
        model_names = self._get_models().keys()
        message = f"""
        ModelTrainer input:
            {self.MERGED_PROCESSED_WON} 
        ModelTrainer output:
            - Training results
            - Preprocessor and Models
        Availaible Models:
            - {list(model_names)}
        """
        print(message)
    
    def _get_models(self, lasso_alpha=5, subset=None):
        n_neighbors = 12
        if self.models == None:
            xgb = XGBRegressor(objective="reg:linear", verbosity=0,random_state=self.random_state)
            cat = CatBoostRegressor(verbose=0,random_state = self.random_state)
            max_iter = 1000
            self.models = {
                "knn_regressor": make_pipeline(
                        StandardScaler(),
                        KNNImputer(n_neighbors=n_neighbors),
                        StandardScaler(),
                        KNeighborsRegressor(n_neighbors= 10), 
                    ),
                "lg_regressor": make_pipeline (
                    StandardScaler(),
                    KNNImputer(n_neighbors=n_neighbors),
                    StandardScaler(),
                    LogisticRegression(random_state=self.random_state),
                    ),
                "xgb_regressor": make_pipeline (
                    StandardScaler(),
                    KNNImputer(n_neighbors=n_neighbors),
                    StandardScaler(),
                    XGBRegressor(objective="reg:linear", verbosity=0, random_state=self.random_state),
                ),
                "cat_regressor": make_pipeline (
                    StandardScaler(),
                    KNNImputer(n_neighbors=n_neighbors),
                    StandardScaler(),
                    CatBoostRegressor(verbose=0, random_state = self.random_state),
                ),
                "gb_regressor": make_pipeline ( 
                    StandardScaler(),
                    KNNImputer(n_neighbors=n_neighbors),
                    StandardScaler(),
                    GradientBoostingRegressor(max_depth = 3, random_state=self.random_state),
                ),
                "rf_regressor": make_pipeline ( 
                    StandardScaler(),
                    KNNImputer(n_neighbors=n_neighbors),
                    StandardScaler(),                           
                    RandomForestRegressor(max_depth = 3, random_state=self.random_state),
                ),
                "ensemble_regressor": make_pipeline( 
                    StandardScaler(),
                    KNNImputer(n_neighbors=n_neighbors),
                    StandardScaler(),
                    VotingRegressor([('xgb', XGBRegressor(objective="reg:linear", verbosity=0, gamma=0.2,max_depth = 5,learning_rate=0.001,subsample = 0.1, random_state=self.random_state)), 
                                                       ('cat', CatBoostRegressor(verbose=0,learning_rate=0.1, l2_leaf_reg=7,iterations=500,depth=10,border_count=20,random_state = self.random_state)),
                                                       ('gb',GradientBoostingRegressor(max_depth = 3, n_estimators=500,min_samples_split=0.5,min_samples_leaf=0.2,learning_rate=0.01,random_state=self.random_state))]),
                ),
                "lasso_knn_regressor": make_pipeline(
                        StandardScaler(),
                        KNNImputer(n_neighbors=n_neighbors),
                        StandardScaler(),
                        SelectFromModel(Lasso(alpha=lasso_alpha, max_iter=max_iter)),
                        KNeighborsRegressor(n_neighbors= 10),
                ), 
                "lasso_lg_regressor": make_pipeline(
                        StandardScaler(),
                        KNNImputer(n_neighbors=n_neighbors),
                        StandardScaler(),
                        SelectFromModel(Lasso(alpha=lasso_alpha, max_iter=max_iter)),
                        LogisticRegression(random_state=self.random_state),
                ), 
                "lasso_xgb_regressor": make_pipeline(
                        StandardScaler(),
                        KNNImputer(n_neighbors=n_neighbors),
                        StandardScaler(),
                        #IterativeImputer(random_state=0, max_iter=1,verbose=2,estimator = RandomForestRegressor()),
                        SelectFromModel(Lasso(alpha=lasso_alpha, max_iter=max_iter)),
                        XGBRegressor(objective="reg:linear", verbosity=0, random_state=self.random_state),
                ), 
                "lasso_cat_regressor": make_pipeline(
                        KNNImputer(n_neighbors=n_neighbors),
                        StandardScaler(),
                        SelectFromModel(Lasso(alpha=lasso_alpha, max_iter=max_iter)),
                        CatBoostRegressor(verbose=0, random_state = self.random_state),
                ),
                "lasso_gb_regressor": make_pipeline(
                        StandardScaler(),
                        KNNImputer(n_neighbors=n_neighbors),
                        StandardScaler(),
                        SelectFromModel(Lasso(alpha=lasso_alpha, max_iter=max_iter)),
                        GradientBoostingRegressor(max_depth = 3, random_state=self.random_state),
                ),
                "lasso_rf_regressor": make_pipeline(
                        StandardScaler(),
                        KNNImputer(n_neighbors=n_neighbors),
                        StandardScaler(),
                        SelectFromModel(Lasso(alpha=lasso_alpha, max_iter=max_iter)),
                        RandomForestRegressor(max_depth = 3, random_state=self.random_state),
                ),    
                "lasso_ensemble_regressor": make_pipeline(
                        StandardScaler(),
                        KNNImputer(n_neighbors=n_neighbors),
                        StandardScaler(),
                        SelectFromModel(Lasso(alpha=lasso_alpha, max_iter=max_iter)),
                        VotingRegressor([('xgb', xgb), ('cat', cat)]),
                )         
                
            }
        if subset == None:
            return self.models
        elif len(subset) == 0 :
            return self.models
        else:
            subset_models = {key: self.models[key] for key in subset}
            return subset_models

    def _get_scoring(self,num_of_columns, num_of_columns_transformed):
        if self.scoring == None:
            adj_r2_scorer = make_scorer(adj_r2_score, greater_is_better=True, n=num_of_columns, p=num_of_columns)
            self.scoring = {
                "r2": "r2",
                # "adj_r2": adj_r2_scorer,
                "sklearn MAPE": "neg_mean_absolute_percentage_error",
                "neg_root_mean_square_error": "neg_root_mean_squared_error",
                "neg_mean_squared_error": "neg_mean_squared_error",
                }
        
        return self.scoring
    
    def train(self):
        return self.trainByName([])
    
    def trainByName(self, model_names=None):
        models_dict = None
        if isinstance(model_names, list):
            models_dict = self._get_models(subset=model_names)
        else:
            raise TypeError(f'{model_names} must be a list')
        return self.trainByDict(models_dict)
    
    def trainByDict(self, models,feature_importance = False):
        results_dict = {}
        preprocessor = self._get_preprocessor()
        self._impute_missing(method='none')
        self._train_test_split()
        scoring = self._get_scoring(self.df_won.shape[0],len(self.X_train.columns))
        
        for model_name, model in models.items():
            pipe = make_pipeline(preprocessor, model)
            self.progressNext(f'ModelTrainer: training {model_name}')
            results_dict[model_name] = mean_std_cross_val_scores(pipe, 
                                                            self.X_train, 
                                                            self.y_train, 
                                                            model_scoring=scoring, 
                                                            std_dev=False,
                                                            return_train_score=True,)
                
        self.progressNext(f'ModelTrainer: training finished, evaluating against test data')
        
        eval_results_dict = {}
        lasso_dict = {}
        for model_name, model in models.items():
            pipe = make_pipeline(preprocessor, model)
            pipe.fit(self.X_train,self.y_train)
            if feature_importance:
                model_instance = model.named_steps['xgbregressor']
                # Get the feature importance values
                feature_importance = model_instance.feature_importances_
                column_transformer = pipe.named_steps['columntransformer']
                feature_names = column_transformer.transformers_[0][2]  # Assuming the first transformer is for numeric features
                # Create a dictionary mapping feature names to importance values
                feature_importance_dict = dict(zip(feature_names, feature_importance))
                # Sort the dictionary by importance values in descending order
                sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
                # Print the top 20 features
                print(f"\nFeature importance (top 20) for model: {model_name}")
                for feature, importance in sorted_features[:20]:
                    print(f"  Feature: {feature}, Importance: {importance * 100:.2f}%")
                save_model(preprocessor,os.path.join(os.getcwd(),".","models","preprocessor"))
                
            self.progressNext(f"ModelTrainer: evaluating '{model_name}' against test data")
            eval_results_dict[model_name] = evaluate_model(pipe,self.X_test,self.y_test)
            # If the model uses lasso feature selection, count the number of selected features
            if 'lasso' in model_name:
                # Get the SelectFromModel instance from the pipeline
                # Get the model (or secondary pipeline) from the primary pipeline
                lasso_pipe = pipe.named_steps[list(pipe.named_steps.keys())[-1]]
                # Get the SelectFromModel instance from the secondary pipeline
                selector = lasso_pipe.named_steps['selectfrommodel']
                # Get the support mask
                mask = selector.get_support()
                # Count the selected features
                num_features = np.sum(mask)
                lasso_dict[model_name] = num_features
            
        self.results_df = pd.concat([pd.DataFrame(results_dict).T, pd.DataFrame(eval_results_dict).T], axis=1)
        self.progressNext(f'ModelTrainer: training and evaluation finished, see results') 
        return self.results_df, lasso_dict
    
    def trainNewPipe(self, pipe, model_name):
        results_dict = {}
        models = None
        if not isinstance(pipe, Pipeline):
            raise TypeError(f'{model_name} must be a Pipeline')
        self._impute_missing(method='none')
        self._train_test_split()
        scoring = self._get_scoring(self.df_won.shape[0],len(self.X_train.columns))
        self.progressNext(f'ModelTrainer: training {model_name}')
        results_dict[model_name] = mean_std_cross_val_scores(pipe, 
                                                            self.X_train, 
                                                            self.y_train, 
                                                            model_scoring=scoring, 
                                                            std_dev=False,
                                                            return_train_score=True)
        self.progressNext(f'ModelTrainer: training finished, evaluating against test data')
        eval_results_dict = {}
        lasso_dict = {}
        pipe.fit(self.X_train,self.y_train)
        self.progressNext(f"ModelTrainer: evaluating '{model_name}' against test data")
        eval_results_dict[model_name] = evaluate_model(pipe,self.X_test,self.y_test)
        if 'lasso' in model_name:
            lasso_pipe = pipe.named_steps[list(pipe.named_steps.keys())[-1]]
            selector = lasso_pipe.named_steps['selectfrommodel']
            mask = selector.get_support()
            num_features = np.sum(mask)
            lasso_dict[model_name] = num_features
        self.results_df = pd.concat([pd.DataFrame(results_dict).T, pd.DataFrame(eval_results_dict).T], axis=1)
        self.progressNext(f'ModelTrainer: training and evaluation finished, see results')
        return self.results_df, lasso_dict
    
    def tunePipe(self,model_name,param_grid, n_iter = 100):
        print('this method must be called after train()')
        # Define the scoring method
        scoring = make_scorer(mape_score, greater_is_better=False)
        model_to_tune = self.getModel(model_name)
        if (model_to_tune != None):
            # create a RandomizedSearchCV object
            preprocessor = self._get_preprocessor()
            pipe = make_pipeline(preprocessor, model_to_tune)
            random_search = RandomizedSearchCV(estimator=pipe, 
                                            param_distributions=param_grid, 
                                            scoring=scoring, cv=5, n_jobs=-1,
                                            n_iter=n_iter,
                                            verbose=0)
            random_search.fit(self.X_train, self.y_train)
            best_model = random_search.best_estimator_
            best_params = random_search.best_params_
            return best_model, best_params
        else:
            print (f'cannot find model {model_name}')
            return None, None
    
    
    def getModel(self,model_name):
        """
        Given a model's name as a string, this method returns the corresponding model.
        """
        if model_name in self.models:
            return self.models[model_name]
        else:
            print(f"No model named {model_name} found.")
            return None
        

    def getResults(self):
        if self.results_df == None:
            print('Results are empty, training first ...')
            self.train()
        else:
            return self.results_df
        
if __name__ == '__main__':
    notebook_dir = os.getcwd()
    data_dir = os.path.join(notebook_dir, ".", "data")
    model_trainer_ml = ModelTrainerML(data_dir)
    # model_trainer_ml.printHelp()
    model_trainer_results, lasso_dict = model_trainer_ml.trainByName([
                                                         'lasso_knn_regressor', 
                                                         'lasso_lg_regressor', 
                                                         'lasso_xgb_regressor', 
                                                         'lasso_cat_regressor',
                                                         'lasso_gb_regressor',
                                                         'lasso_rf_regressor',
                                                         'lasso_ensemble_regressor'
                                                         ])
    print(model_trainer_results)