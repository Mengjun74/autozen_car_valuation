import os
import mysql.connector
import json
import pandas as pd
from progress.bar import ChargingBar
from autozen_features import AutozenFeatures
from sklearn.impute import KNNImputer
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import torch
from utils import mape

VERBOSE = False 

class PipelineStep:
    """
    PipelineStep provides basic functionalities to progress through the steps 
    of a data pipeline and manage a database connection.
    """
    def __init__(self, data_dir, db_creds=None,num_of_steps=1,progress_message="Progress msg here ..."):
        """
        Initialize the object and establish a database connection.
        Parameters:
            data_dir: Directory for the data files
            db_creds: Credentials for the database connection (optional)
            num_of_steps: Total number of pipeline steps (default: 1)
            progress_message: Message to display in the progress bar
        """
        self.data_dir = data_dir
        self.db_creds = self.get_db_creds() if db_creds is None else db_creds
        self.progress_bar = ChargingBar(progress_message, max=num_of_steps)
        self.cnx=None
    
    def get_db_creds(self):
        """
        Get database credentials from environment variables or a credentials file.
        Returns: dictionary of database credentials
        """
        db_creds = {k: os.getenv(f'AUTOZEN_DB_{k.upper()}') for k in ['user', 'password', 'host', 'database']}
        if all(db_creds.values()):
            return db_creds

        with open(os.path.join("autozen-credentials.json"), 'r') as file:
            return json.load(file)
        
    def progressNext(self, status_message):
        """
        Increment the progress bar and print a status message.
        Parameters:
            status_message: The message to print
        """
        self.progress_bar.suffix = status_message
        self.progress_bar.next()
        print(f'\n{status_message}')
        
    def disconnectDB(self):
        """Closes the database connection if it's open."""
        if self.cnx is not None:
            self.cnx.close()
            self.cnx = None

    def reconnectDB(self):
        """Establishes a new database connection if it's closed."""
        if self.cnx is None:
            self.cnx = mysql.connector.connect(**self.db_creds)
    
    @staticmethod
    def save_nn_model(model, model_file_name):
        """
        Saves a PyTorch model's state dictionary to a file.
        
        Parameters:
            model: The PyTorch model to save
            model_file_name: The name of the file to save the model to
        """
        torch.save(model.state_dict(), model_file_name)
    
    
    @staticmethod
    def trainer_regression(model, criterion, optimizer, 
                           trainloader, validloader, 
                           device=None, epochs=5,
                           l1_lambda=0.01):
        """
        Trains a regression model and records loss and score at each epoch.

        Parameters:
        model: The model to train
        criterion: The loss function
        optimizer: The optimization algorithm
        trainloader: Data loader for the training set
        validloader: Data loader for the validation set
        device: The device (CPU or GPU) to use for training
        epochs: The number of training epochs (default: 5)
        l1_lambda: The lambda for L1 regularization (default: 0.01)

        Returns:
        A dictionary with the training and validation loss and scores.
        """
        train_loss, valid_loss, valid_mape, train_mape, valid_r2, train_r2 = [], [], [], [], [], []
        count = 0
        for epoch in range(epochs):
            train_batch_loss = 0
            train_batch_mape = 0
            train_batch_r2 = 0
            valid_batch_loss = 0
            valid_batch_mape = 0
            valid_batch_r2 =0
            
            for X, y in trainloader:
                X, y = X.to(device), y.to(device) 
                optimizer.zero_grad()       # Clear gradients w.r.t. parameters
                y_hat = model(X).flatten()  # Forward pass to get output
                loss = criterion(y_hat, y)  # Calculate loss
                # Add L1 regularization
                if  l1_lambda != -1:
                    l1_lambda = 0.001
                    l1_norm = sum(p.abs().sum() for p in model.parameters())
                    loss = loss + l1_lambda * l1_norm
                loss.backward()             # Getting gradients w.r.t. parameters
                optimizer.step()            # Update parameters
                train_batch_loss += loss.item()       # Add loss for this batch to running total
                train_batch_mape += mape(y_hat,y)
                # Convert the predicted values and the actual values to NumPy arrays
                numpy_predictions = y_hat.detach().cpu().numpy()
                numpy_y = y.cpu().numpy()
                train_batch_r2 += r2_score(numpy_predictions, numpy_y)
            train_loss.append(train_batch_loss / len(trainloader)) # mse
            train_mape.append(train_batch_mape / len(trainloader)) # mape
            train_r2.append(train_batch_r2/len(trainloader)) # r2
            
            # Validation
            with torch.no_grad():  # this stops pytorch doing computational graph stuff under-the-hood and saves memory and time
                for X, y in validloader:
                    X, y = X.to(device), y.to(device) 
                    y_hat = model(X).flatten()
                    loss = criterion(y_hat, y)
                    valid_batch_loss += loss.item()       # Add loss for this batch to running total
                    valid_batch_mape += mape(y_hat,y)
                    numpy_predictions = y_hat.detach().cpu().numpy()
                    numpy_y_test = y.cpu().numpy()
                    valid_batch_r2 += r2_score(numpy_predictions, numpy_y_test)
            valid_loss.append(valid_batch_loss / len(validloader)) # mse
            valid_mape.append(valid_batch_mape / len(validloader))  # mape
            valid_r2.append(valid_batch_r2/len(validloader)) # r2
            
            # Print progress                
            if count % 5 == 0 or count == (epochs -1) :
                print(f"Epoch {epoch + 1}: ",
                    f"Train Loss: {train_loss[-1]:.3f}. ",
                    f"Valid Loss: {valid_loss[-1]:.3f}. ",
                    f"Train MAPE: {train_mape[-1]:.2f}. ",
                    f"Valid MAPE: {valid_mape[-1]:.2f}. ",
                    f"Train R2: {train_r2[-1]:.2f}. ",
                    f"Valid R2: {valid_r2[-1]:.2f}. ")
            count += 1
        
        results = {"train_loss": train_loss,
                "valid_loss": valid_loss,
                "train_mape": train_mape,
                "valid_mape": valid_mape,
                "train_r2":train_r2,
                "valid_r2":valid_r2}
        return results