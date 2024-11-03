# CTR Prediction - GBDT+LR

This project implements a binary classification prediction system based on a custom LightGBM model combined with Logistic Regression. The system provides a complete machine learning workflow, including data preprocessing, model training, and prediction.


## File Descriptions

- **data/**: Contains the datasets.
  - `data.csv`: The combined dataset containing both training and testing data.
  - `train.csv`: The training dataset used for model training.
  - `test.csv`: The test dataset used for model prediction.

- **data_preprocess/**: Contains code related to data preprocessing.
  - `data_preprocess.py`: Includes functions for data preprocessing, responsible for reading and processing raw data.

- **model/**: Contains the model implementation.
  - `LightGBM.py`: Custom implementation of the LightGBM model, which can be modified according to user needs.

- **predict/**: Contains prediction-related code.
  - `gbdt_lr_predict.py`: The prediction function that combines GBDT and Logistic Regression, responsible for model training and prediction processes.

- **main.py**: The main function, responsible for orchestrating data preprocessing, model training, and prediction.

- **main.ipynb**: Jupyter Notebook file that provides usage examples and references for understanding and using the model.

