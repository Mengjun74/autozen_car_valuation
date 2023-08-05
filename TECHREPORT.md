# Technical Report: Exploring Different Models and Methods in Data Science to predict used car prices

## Table of Contents
- [Technical Report: Exploring Different Models and Methods in Data Science to predict used car prices](#technical-report-exploring-different-models-and-methods-in-data-science-to-predict-used-car-prices)
  - [Table of Contents](#table-of-contents)
  - [1. Introduction](#1-introduction)
  - [2. Background](#2-background)
  - [3. Methodology](#3-methodology)
  - [4. Findings and Analysis](#4-findings-and-analysis)
  - [5. Unsuccessful Approaches](#5-unsuccessful-approaches)
  - [6. Final Thoughts and Future Recommendations](#6-final-thoughts-and-future-recommendations)
  - [7. References](#7-references)
  - [8. Appendices](#8-appendices)

<a name="introduction"></a>
## 1. Introduction

Autozen, an online marketplace for used cars, has streamlined and simplified the process of selling used cars. The Autozen Valuation Guru project was initiated, however, in recognition of the need for a more efficient car valuation process. The goal of the project was to improve the precision and dependability of used car valuations by employing advanced Data Science and Machine Learning techniques.

The purpose of this report is to provide a comprehensive overview of the project, including the problem definition, objectives, methods used, final data product, and recommendations for future improvements.

The approach to achieving the project's objectives is multifaceted. Initially, we assessed the precision and efficacy of Autozen's current valuation method. To improve the accuracy of valuations, a novel predictive model that included the vehicle's inspection report was developed after this evaluation. Lastly, we sought to produce a more precise prediction interval for auction prices, thereby reducing the uncertainty surrounding the estimated values. This required extensive steps of data exploration, preprocessing, model training, optimisation, and validation.


<a name="background"></a>
## 2. Background

Autozen operates an online marketplace for used cars that simplifies the selling process. However, one difficulty is accurately and reliably estimating the value of used automobiles. The existing valuation system relies on the Canadian Black Book (CBB) to estimate the price range, which has shown discrepancies and limitations, primarily due to various factors affecting the auction prices of used cars, such as mileage, condition, and damages. Therefore, Autozen recognised the need for an improved car valuation process, which led to the development of the Autozen Valuation Guru project.

Autozen requires precise vehicle valuations for multiple reasons. It allows customers to receive a fair price for their used vehicles, fosters customer confidence, and enables authorised dealerships to submit competitive and reasonable bids. Moreover, more accurate valuations are advantageous for Autozen, as they allow the company to optimise its decision-making process during auctions, thereby improving the effectiveness of their business model.

Let's begin with the fundamentals in order to comprehend the data science concepts involved in this endeavour. Data science entails, at its core, the extraction of knowledge and insights from structured or unstructured data. We focused primarily on structured data for this project, including vehicle make, model, year, mileage, and other inspection-related details. Due to limited time and resources, all other unstructured elements, including notes and comments, photos, and commercial documents (such as Carfax reports), were disregarded.

Machine Learning (ML) is one of the primary data science techniques employed in this article. ML is the process of teaching a computer to discover patterns in data, which it can then use to make predictions or decisions without being explicitly programmed to do so. For example, we used ML models to predict the auction prices of used cars based on a variety of car characteristics.

Preprocessing is another important concept that prepares raw data for machine learning models. This may involve handling missing data, standardising numerical data, and encoding categorical data, all of which improve the data's suitability for ML algorithms.

Model training is the process of teaching an ML model to recognise data patterns. We then evaluate the performance of the model by applying it to data that it has never seen. The objective is to create a model that generalises well, or performs well not only on the training data, but also on new, unseen data.

Model optimisation and regularisation entail fine-tuning the ML model to enhance its performance and prevent overfitting, which occurs when the model learns the training data too well and performs poorly on new data.


<a name="methodology"></a>
## 3. Methodology

Several machine learning (ML) models were explored to predict the winning bids on used cars. These models were selected based on their ability to handle both **categorical** and **numerical** data, their performance in regression tasks, and their capability to address missing data and feature interactions. The primary models utilised include Logistic Regression, KNeighborsRegressor, XGBRegressor, CatBoostRegressor, GradientBoostingRegressor and RandomForestRegressor.

1. **Logistic Regression:** (base model) This model typically suits binary classification tasks but served as a baseline for comparison in this project. It uses a logistic function to model a binary dependent variable and can be extended to handle multi-class classification and even regression problems. Its simplicity and interpretability make it a good starting point.

2. **KNeighborsRegressor:** This model was chosen for its ability to handle non-linear relationships and complex patterns in the data. It works by finding the 'k' nearest examples to a given input and averaging their values to form a prediction. Despite being simple to implement, it can be computationally demanding for larger datasets (greater than 100,000 observations, which is not our case here).

3. **XGBRegressor, CatBoostRegressor, and GradientBoostingRegressor:** These are gradient-boosting algorithms, which create a strong predictive model by combining multiple weak models (typically decision trees), aiming to minimize the residual errors gradually. These models were selected due to their demonstrated effectiveness in regression tasks and their capabilities to handle both numerical and categorical features, missing data, and feature interactions.

4. **RandomForestRegressor:** An ensemble model combining multiple decision trees to produce more robust predictions. It has the advantage of addressing feature interactions automatically and offering estimates of feature importance.

5. **Deep Neural Networks:** These models are highly flexible and powerful predictors that can capture complex patterns and relationships in data. Neural networks are composed of layers of nodes or "neurons", and they learn to map inputs to outputs through a process of weighted connections and non-linear transformations. However, they can be computationally intensive, require careful tuning, and their results are not always easily interpretable.

For applying these models, the following steps were taken:

1. **Data Preprocessing:** The categorical, ordinal, numerical, and binary features were encoded using OneHotEncoder, OrdinalEncoder, and StandardScaler. This made the data more suitable for ML algorithms.

2. **Addressing Missing Data:** Various techniques, including visual inspection and the construction of a missing data matrix, were used to understand the missingness in the data. Following that, KNN and Iterative imputers were used to impute the missing data, which was deemed Missing Not At Random (MNAR). MNAR refers to a situation in a dataset where the missingness of data is dependent on the values of the missing data itself.

3. **Model Training:** Each model was trained using this prepared training dataset. The training process involved feeding the models with inputs (used car features) and expected outputs (winning bids) to learn the relationship between them.

4. **Model Evaluation:** The trained models were then evaluated on a separate testing dataset using the Mean Absolute Percentage Error (MAPE) as the performance metric. MAPE is a measure of prediction accuracy in statistics, representing the average of the absolute percentage differences between the actual and predicted values.

5. **Model Optimization and Regularization:** To enhance model performance, RandomizedSearchCV was employed for hyperparameter tuning, and lasso regularization was used for feature selection and overfitting prevention. Hyperparameter tuning is the process of selecting the optimal parameters in a machine learning model to improve its performance on a given dataset. Overfitting in machine learning refers to a situation where a model learns the training data so well that it performs poorly on unseen test data due to its inability to generalize from the training set to new data.

The detailed scripts for data preprocessing, missing data handling, model training, evaluation, and optimization can be found in the associated GitHub repository under the **scripts** directory. The repository includes Jupyter notebooks, Python scripts, and detailed README files explaining each step in detail, and getting started instructions to reproduce all the results mentioned in the final report.


<a name="findings"></a>
## 4. Findings and Analysis

In the pursuit of an accurate and reliable tool for the valuation of used cars, our project has been successful in providing Autozen with a data product that predicts auction prices based on existing used card data in addition to the inspection information collected by Autozen specialists. This mitigates the reliance on the Canadian Black Book (CBB) valuation method and enables both sellers and buyers to engage in informed transactions with fair estimates.

**Strengths**

- **Accuracy:** The data product has exhibited a strong performance in predicting the auction prices, which has been validated through various machine learning models. The best model, `'**XGBoostRegressor**' achieved a MAPE score of 3% on unseen data.

- **Transparency:** The tool took advantage of car inspection data for the prediction, making the process transparent and trustworthy for users.

- **Continuous Improvement:** The reproducible data pipeline designed as part of this project allowed for ongoing refinement and improvement of the data product.

**Limitations and Challenges**

Despite its robust performance, the data product also has some limitations that need to be considered:

- The prediction accuracy heavily relies on the quality and completeness of the car inspection data, which presents a challenge if the data is missing or inaccurate.

- The data collection and preprocessing are not straightforward and require an intimate understanding of the internal data structures used by the web portal. In the future, we recommend that Autozen have a background process that creates the training data on an ongoing basis, as they are the domain experts of their internal information model.

- Ethical concerns, such as potential biases in the data or implications for pricing fairness, need to be closely monitored and managed to ensure ethical and fair practises.

These findings highlight both the effectiveness of our approach in addressing Autozen's needs and the areas that require further improvement. For detailed code, model outputs, and data visualisations, please refer to the relevant notebooks and scripts in the [README.md](README.md).


<a name="unsuccessful"></a>
## 5. Unsuccessful Approaches

Despite the success with the models mentioned in the previous section, there were a few approaches that did not perform as expected. Understanding these unsuccessful strategies is just as important as identifying successful ones, as it gives insights into the characteristics of the problem at hand and aids in refining our methodology.

1. **Logistic Regression:** (base model) Initially, a basic linear regression model was tested due to its simplicity and interpretability. However, it did not account for non-linear relationships between features and the target variable, resulting in poor performance on both the training and testing sets. MAPE scores over 30% which are worse than the existing scores of 8-10% by Autozen.

2. **KNeighborsRegressor:** This model, while effective for small datasets or problems with clear boundaries, didn't provide satisfactory results for our problem. The high dimensionality of the data led to the curse of dimensionality, where the increased distance between data points adversely affected the model's performance. MAPE scores over 30% which are worse than the existing scores of 8-10% by Autozen.

3. **Deep Neural Networks:** The complexity of deep neural networks initially seemed suitable for handling the intricacies of this dataset. However, these models proved challenging to tune and computationally expensive. Moreover, their 'black-box' nature made the interpretability of the results difficult, which is an important aspect for Autozen. MAPE scores were around 8% which matched the existing outcomes that Autozen is currently getting, however, even with 1000 epochs, various dropout rates, and L1 regularization, we were not able to go below that MAPE score for test data. An epoch in neural networks refers to one complete pass of the entire dataset through the network during training.

Each of these unsuccessful methods was discontinued due to their unsatisfactory performance and other challenges faced during implementation. Detailed documentation, including code and comments, can be found in the accompanying Jupyter notebooks in the project repository ([README.md](README.md)).

Through these unsuccessful approaches, valuable insights were gained. The complexity and non-linearity of the problem were highlighted, underscoring the need for models capable of capturing intricate feature interactions and patterns. Additionally, these experiences reinforced the importance of model interpretability in this context, emphasising the need to balance prediction accuracy with interpretability and computational efficiency. This is why we pursued the gradient-boosting algorithms to further fine-tune the winning model, **XGBRegressor**.


<a name="final"></a>
## 6. Final Thoughts and Future Recommendations
The Autozen Valuation Guru project has been a journey of exploration, learning, and growth. Through the use of various machine learning models, we successfully developed a predictive model that provides accurate valuations of used cars based on their inspection data. Key findings indicate that gradient-boosting algorithms (XGBRegressor, CatBoostRegressor, and GradientBoostingRegressor) have proven to be the most effective models for this task. Deep Neural Networks require extensive computational resources and time for tuning and training, making them unsuitable for this project, at this point in time.

The developed data product provides Autozen with a valuable tool to aid in transparent and informed transactions. By making predictions based on inspection data alone, it enhances what Autozen was getting from the Canadian Black Book (CBB) valuation method, enabling both buyers and sellers to have a fair and reliable estimate.

In terms of future work, the exploration of other machine learning models, such as Support Vector Machines (SVMs) or more complex neural network architectures, could be beneficial. Furthermore, the adoption of advanced techniques like ML Operations (MLOps) could potentially streamline the model development and prediction process when used in production.

Moreover, efforts should be made to address the limitations of the current data product, including improving the completeness and quality of car inspection data and careful monitoring for potential biases to ensure fairness.

This project serves as a solid foundation for future team members to build on. The lessons learned from both successful and unsuccessful approaches provide valuable insights that can guide future data science endeavours at Autozen. With continuous improvement and refinement, Autozen's valuation process can achieve even greater accuracy, reliability, and fairness, solidifying its competitive edge in the used car market.

<a name="references"></a>
## 7. References
* [README](README.md)
* [Autozen.ipynb](Autozen.ipynb)
* [EDA.ipynb](EDA.ipynb)
* [Data Schema](documents/Appendix_A_DataSchema.md)


<a name="appendices"></a>
## 8. Appendices
| Term | Definition |
|-------------------------|------------|
| Training | The process of learning patterns in data or learning the parameters of a model. |
| Test Data | The subset of data used to evaluate the performance of a trained model, this data is not used for training the model. |
| Training Data | The subset of data used to train a model. |
| Validation Data | The subset of data used to fine-tune model parameters and to prevent overfitting. |
| Overfitting | When a model learns too well from the training data and performs poorly on unseen data. |
| Underfitting | When a model cannot learn well enough from the training data and performs poorly on both training and unseen data. |
| Supervised Learning | A type of machine learning where the model learns from labelled training data to make predictions. |
| Deep Learning | A subset of machine learning that uses artificial neural networks with multiple layers (deep networks). |
| Epoch | One complete pass through the entire training dataset during the training of a machine learning model. |
| Batch Size | The number of training examples used in one iteration of model training. |
| Gradient Descent | An optimization algorithm used to minimize the loss function by iteratively adjusting the model's parameters. |
| Learning Rate | The step size at each iteration while moving towards a minimum of a loss function in gradient descent. |
| Regularization | A technique to prevent overfitting by adding a penalty term to the loss function. |
| Loss Function | A method of evaluating how well a machine learning model is performing during training. |
| Hyperparameter | A parameter whose value is set before the learning process begins, used to control the learning process. |
| Parameter | A variable internal to the model that the algorithm learns from the training data. |
| Feature | A variable in the dataset used as input for a machine learning model. |
| Target Variable/Label | The variable that a machine learning model aims to predict. |
| Feature Engineering | The process of creating new features or modifying existing features to improve model performance. |
| Bias-Variance Tradeoff | The balance that must be achieved between bias (underfitting) and variance (overfitting). |
| Cross-Validation | A technique for assessing how well a model will generalize to an independent dataset by partitioning the data into subsets. |
| Ensemble Methods | Techniques that create multiple models and then combine them to produce improved results. |
| MAPE | Mean Absolute Percentage Error. It represents the average of the absolute percentage differences between the actual and predicted values.|
