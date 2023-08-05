# # model_nn_training.py
from base_step import PipelineStep
from base_step import VERBOSE
import os
import pandas as pd
import numpy as np
from progress.bar import ChargingBar
from autozen_features import AutozenFeatures

import torch
from torchsummary import summary
from torch import nn
from torchvision import transforms, datasets, utils
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import cross_validate, train_test_split, cross_val_score
from sklearn.compose import make_column_transformer
from feature_engineering import FeatureEngineer



class ModelTrainerNN (FeatureEngineer):
    
    """
    The ModelTrainerNN class extends the FeatureEngineer class and is responsible for training machine learning models
    using PyTorch and saving the trained models for later use.
    """
    def __init__(self, data_dir,random_state=591,test_size=0.2,num_of_steps=4):
        super().__init__(data_dir=data_dir, db_creds=None,num_of_steps=num_of_steps,
                         progress_message='ModelTrainerNN: processing...', 
                         random_state=random_state,test_size=test_size)
        if torch.cuda.is_available():
            device_name = 'cuda'
        elif torch.backends.mps.is_available():
            device_name = 'mps'
        else:
            device_name = 'cpu'
        self.device = torch.device(device_name)
        # input
        self.EPOCHS = 100
        self.BATCH_SIZE = 50
        self.LEARNING_RATE = 0.1
        # output
        self.models = None
        self.results_df = None

    
    def printHelp(self):
        message = f"""
        ModelTrainer input:
            {self.MERGED_PROCESSED_WON} 
        ModelTrainer output:
            - Training results
            - Preprocessor and Models
        Availaible Models:
            - ['AutozenNN_A','AutozenNN_B']
        """
        print(message)

    def _get_model(self, input_size, model_name, drop_out):
        self.models = {
            "AutozenNN_A": AutozenNN_A(input_size=input_size,output_size=1),
            "AutozenNN_B": AutozenNN_B(input_size=input_size,drop_out=drop_out,output_size=1),
        }
        model = self.models[model_name]
        return model
    
    def train(self,model_name, drop_out=0.5,on_device=True,EPOCHS=None,BATCH_SIZE=None):
        results_df = None
        if EPOCHS == None:
            USED_EPOCHS = self.EPOCHS
        else:
            USED_EPOCHS = EPOCHS
        if BATCH_SIZE == None:
            USED_BATCH_SIZE = self.BATCH_SIZE
        else:
            USED_BATCH_SIZE = BATCH_SIZE 
        # preprocessor
        preprocessor = self._get_preprocessor()
        self._impute_missing(method='knn:mnar')
        self._train_test_split()
        # preprocessor
        preprocessor = self._get_preprocessor()
        # preprocessor.fit(self.X_train)
        preprocessor.fit(self.X)
        X_train_transformed = torch.tensor(preprocessor.transform(self.X_train)).float()
        y_train_tensor = torch.tensor(self.y_train.values).float()  # convert y_train to a tensor
        X_test_transformed = torch.tensor(preprocessor.transform(self.X_test)).float()
        y_test_tensor = torch.tensor(self.y_test.values).float()  # convert y_train to a tensor
        dataset = TensorDataset(X_train_transformed, y_train_tensor)  # use the tensor version of y_train
        dataloader = DataLoader(dataset, batch_size=USED_BATCH_SIZE, shuffle=True)
        dataset_valid = TensorDataset(X_test_transformed, y_test_tensor)  # use the tensor version of y_test
        dataloader_valid = DataLoader(dataset_valid, batch_size=USED_BATCH_SIZE, shuffle=True)
        XX, yy = next(iter(dataloader))
        if VERBOSE:
            print(f"Shape of feature data (X) in batch: {XX.shape}")
            print(f"Shape of response data (y) in batch: {yy.shape}")
            print(f"Total number of batches: {len(dataloader)}")
        
        model = self._get_model(input_size=X_train_transformed.shape[1], model_name=model_name, drop_out=drop_out)
        if on_device:
            model.to(self.device)
        if VERBOSE:
            print(f'{model_name}')
            summary(model)
        # To fit the model, we need to define a loss function and an optimizer
        criterion = nn.MSELoss()  # loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=self.LEARNING_RATE)  # optimization algorithm
        self.progressNext(f'ModelTrainer NN: training started')
        if on_device:
            results_df = self.trainer_regression(model, criterion, optimizer, dataloader, dataloader_valid, 
                                    device=self.device, epochs=USED_EPOCHS)
        else:
            results_df = self.trainer_regression(model, criterion, optimizer, dataloader, dataloader_valid, 
                                    device=None, epochs=USED_EPOCHS)
        self.progressNext(f'ModelTrainer NN: training completed')
        return model,results_df
    
    def save_model(self,model_name):
        model = self.getModel(model_name)
        model_file = os.path.join(os.getcwd(), ".", "model",model_name)
        self.save_nn_model(model, model_file)
        
    def load_model(self, model_name):
        model_file = os.path.join(os.getcwd(), ".", "model", model_name)
        model = torch.load(model_file)
        return model

    def getModel(self,model_name):
        """
        Given a model's name as a string, this method returns the corresponding model.
        """
        if model_name in self.models:
            return self.models[model_name]
        else:
            print(f"No model named {model_name} found.")
            return None
        
        
        
        
class AutozenNN_B(nn.Module):
    def __init__(self, input_size, output_size, drop_out):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, 6000),
            nn.LayerNorm(6000),
            nn.ReLU(),
            nn.Linear(6000,2000),
            nn.Dropout(drop_out),
            nn.LayerNorm(2000),
            nn.ReLU(),
            nn.Linear(2000,4000),
            nn.Dropout(drop_out),
            nn.LayerNorm(4000),
            nn.ReLU(),
            nn.Linear(4000,1000),
            nn.Dropout(drop_out),
            nn.LayerNorm(1000),
            nn.ReLU(),
            nn.Linear(1000, output_size)
        )

    def forward(self, x):
        out = self.main(x)
        return out
    
class AutozenNN_A(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, 5000),
            nn.LayerNorm(5000),
            nn.ReLU(),
            nn.Linear(5000,1000),
            nn.LayerNorm(1000),
            nn.ReLU(),
            nn.Linear(1000,256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        out = self.main(x)
        return out
        
if __name__ == '__main__':
    notebook_dir = os.getcwd()
    data_dir = os.path.join(notebook_dir, ".", "data")
    model_trainer_nn = ModelTrainerNN(data_dir=data_dir,random_state=2023,test_size=0.3,num_of_steps=6)
    # model_trainer_nn.printHelp()
    model, results = model_trainer_nn.train('AutozenNN_A',on_device=False,EPOCHS=20,BATCH_SIZE=None)
    model, results = model_trainer_nn.train('AutozenNN_B',on_device=False,EPOCHS=20,BATCH_SIZE=None)