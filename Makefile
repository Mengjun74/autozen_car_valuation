# Makefile for a data science project

# .PHONY is a special target that helps to avoid conflicts with files and targets
# that have the same name, and improves performance
.PHONY: data_collection data_preprocessing feature_engineering model_nn_training model_ml_training run_pipeline clean help

# DATA_DIR specifies the directory where the data and intermediate processing files are stored
DATA_DIR := data

# Target 'collect' runs data collection script
collect: $(DATA_DIR)/.done_collect

# Target 'process' runs data preprocessing script
process: $(DATA_DIR)/.done_process

# The 'collect' target checks if data collection is completed
$(DATA_DIR)/.done_collect:
	@echo "Collecting Data..."
	python scripts/data_collection.py
	@mkdir -p $(DATA_DIR)  # Make the data directory if it doesn't exist
	@touch $@  # Update the timestamp of the '.done_collect' file

# The 'process' target checks if data preprocessing is completed, depending on data collection
$(DATA_DIR)/.done_process: $(DATA_DIR)/.done_collect
	@echo "Preprocessing Data..."
	python scripts/data_preprocessing.py
	@mkdir -p $(DATA_DIR)  # Make the data directory if it doesn't exist
	@touch $(DATA_DIR)/.done_process  # Update the timestamp of the '.done_process' file

# Target 'train_nn' runs the script for training the neural network model, depending on data preprocessing
train_nn: $(DATA_DIR)/.done_process
	@echo "Training Neural Network Model..."
	python scripts/model_nn_training.py

# Target 'train_ml' runs the script for training the machine learning model, depending on data preprocessing
train_ml: $(DATA_DIR)/.done_process
	@echo "Training Machine Learning Model..."
	python scripts/model_ml_training.py

# Target 'optimize' runs the script for otpimizing the best machine learning model
optimize: $(DATA_DIR)/.done_process
	@echo "Optimizing best model..."
	python scripts/model_optimization.py

# Target 'run_pipeline' runs the entire data pipeline, depending on both 'train_nn' and 'train_ml'
run_pipeline: train_nn train_ml
	@echo "Data Pipeline Completed Successfully."

# Target 'clean' cleans up the data directory by removing the '.done_collect' and '.done_process' files
clean:
	@echo "Cleaning up..."
	rm -rf $(DATA_DIR)/.done_collect $(DATA_DIR)/.done_process

# Target 'help' displays available commands
help:
	@echo "Available commands:"
	@echo "  collect        Run the data collection script"
	@echo "  process        Run the data preprocessing script"
	@echo "  train_nn       Train the neural network model"
	@echo "  train_ml       Train the machine learning model"
	@echo "  optimize       Optimize the best model"
	@echo "  run_pipeline   Run the entire data pipeline"
	@echo "  clean          Clean up the data directory"
