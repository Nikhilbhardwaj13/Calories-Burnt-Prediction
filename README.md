# Calories Burnt Prediction using Machine Learning

This repository contains a machine learning project for predicting the number of calories burnt during physical activities. The project is implemented in Python and leverages various machine learning techniques to achieve accurate predictions.

## Project Overview

Accurately predicting the number of calories burnt during physical activities can help individuals monitor their fitness goals more effectively. This project uses a dataset containing various physical attributes and activities to train a machine learning model that predicts calories burnt.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Code Explanation](#code-explanation)
- [Model](#model)
- [Results](#results)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
  
## Dataset

The dataset used in this project contains the following features:
- `User_ID`: Identifier for the user
- `Gender`: Gender of the user (Male/Female)
- `Age`: Age of the user
- `Height`: Height of the user (in cm)
- `Weight`: Weight of the user (in kg)
- `Duration`: Duration of the activity (in minutes)
- `Heart_Rate`: Heart rate of the user during the activity
- `Body_Temperature`: Body temperature of the user (in Celsius)
- `Calories_Burnt`: Target variable, representing the number of calories burnt

The dataset can be found in the `data/` directory.

## Installation

To run this project, you'll need to have Python installed on your machine. You can install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/Nikhilbhardwaj13/calories-burnt-prediction.git
    cd calories-burnt-prediction
    ```

2. **Run Jupyter Notebook**:

    Open the Jupyter Notebook to explore the data analysis and model training steps:

    ```bash
    jupyter notebook notebooks/calories_burnt_prediction.ipynb
    ```

3. **Training the Model**:

    You can train the model by running the script provided below:

    ```python
    # Import necessary libraries
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import joblib

    # Load the dataset
    def load_data(file_path):
        data = pd.read_csv(file_path)
        return data

    # Preprocess the data
    def preprocess_data(data):
        data = data.dropna()
        data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
        return data

    # Engineer features
    def engineer_features(data):
        data['BMI'] = data['Weight'] / ((data['Height'] / 100) ** 2)
        return data

    # Main training function
    def train_model():
        # Load and preprocess data
        data = load_data('data/dataset.csv')
        data = preprocess_data(data)
        data = engineer_features(data)

        # Split data
        X = data.drop(['User_ID', 'Calories_Burnt'], axis=1)
        y = data['Calories_Burnt']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Save model
        joblib.dump(model, 'models/trained_model.pkl')

        # Evaluate model
        y_pred = model.predict(X_test)
        print(f'MAE: {mean_absolute_error(y_test, y_pred)}')
        print(f'MSE: {mean_squared_error(y_test, y_pred)}')
        print(f'R2 Score: {r2_score(y_test, y_pred)}')

    if __name__ == '__main__':
        train_model()
    ```

4. **Evaluate the Model**:

    You can evaluate the model using the script provided below:

## Import necessary libraries
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

## Load the dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

## Preprocess the data
def preprocess_data(data):
    data = data.dropna()
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
    return data

## Engineer features
def engineer_features(data):
    data['BMI'] = data['Weight'] / ((data['Height'] / 100) ** 2)
    return data

## Main evaluation function
def evaluate_model():
    # Load data
    data = load_data('data/dataset.csv')
    data = preprocess_data(data)
    data = engineer_features(data)

    # Split data
    X = data.drop(['User_ID', 'Calories_Burnt'], axis=1)
    y = data['Calories_Burnt']

    # Load model
    model = joblib.load('models/trained_model.pkl')

    # Predict and evaluate
    y_pred = model.predict(X)
    print(f'MAE: {mean_absolute_error(y, y_pred)}')
    print(f'MSE: {mean_squared_error(y, y_pred)}')
    print(f'R2 Score: {r2_score(y, y_pred)}')

if __name__ == '__main__':
    evaluate_model()

## Project Structure

calories-burnt-prediction/
│
├── data/
│   ├── dataset.csv
│
├── notebooks/
│   ├── calories_burnt_prediction.ipynb
│
├── models/
│   ├── trained_model.pkl
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore

## Code Explanation

### 1. Importing Necessary Libraries

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
```

## Model

The project uses a regression model to predict the number of calories burnt. Various models like Linear Regression, Decision Trees, Random Forest, and Gradient Boosting are evaluated to find the best performing model.

## Results

The model's performance is evaluated using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared. Detailed results and analysis can be found in the notebook `notebooks/calories_burnt_prediction.ipynb`.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Create a new Pull Request.

## Acknowledgements

- Special thanks to [Kaggle](https://www.kaggle.com/datasets/fmendes/fmendesdat263xdemos) for providing the dataset.
- Thanks to all contributors for their valuable input and feedback.
