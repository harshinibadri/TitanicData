Here's the documentation for the Titanic dataset classification project using a Random Forest model, outlining the process and what each step in the code performs:

---

# Titanic Survival Prediction Project

## Overview

The goal of this project is to predict the survival status of passengers aboard the Titanic based on various features such as passenger class, sex, age, fare, family size, and embarkation port. The model is trained using the **Random Forest Classifier** to classify whether a passenger survived (1) or did not survive (0). The Titanic dataset, which includes information on passengers such as age, sex, class, and survival, is used to train the model.

## Steps Performed

### 1. **Importing Necessary Libraries**
The following Python libraries were imported to perform data manipulation, machine learning, and model evaluation:
- **pandas**: For reading and manipulating the dataset.
- **sklearn.model_selection**: To split the data into training and testing sets.
- **sklearn.ensemble**: For using the Random Forest Classifier.
- **sklearn.metrics**: To evaluate the model's performance.
- **sklearn.preprocessing**: For encoding categorical variables.
- **sklearn.impute**: For handling missing values.
- **joblib**: To save and load the trained model.

### 2. **Loading the Dataset**
The Titanic dataset is loaded from the file path `"C:/Users/VENUGOPAL BADRI/Downloads/tested.csv"` using `pd.read_csv()`. This dataset contains various features, including the survival status (`Survived`), passenger class (`Pclass`), sex (`Sex`), age (`Age`), fare (`Fare`), family information (`SibSp`, `Parch`), and embarkation port (`Embarked`).

### 3. **Data Preprocessing**

#### Handling Missing Values
- **Age**: Missing values in the `Age` column are handled by filling them with the median value using `SimpleImputer(strategy='median')`.
- **Fare**: Missing values in the `Fare` column are also filled with the median value using a similar imputer.
- **Cabin**: The `Cabin` column is dropped due to excessive missing values, as keeping it would not provide meaningful data for training.

#### Encoding Categorical Variables
- **Sex**: The `Sex` column (with values 'male' and 'female') is encoded into numerical values using **Label Encoding**: 'male' is encoded as 1 and 'female' as 0.
- **Embarked**: The `Embarked` column is one-hot encoded using `pd.get_dummies()`, creating binary columns for the embarkation ports. We drop the first column to avoid the "dummy variable trap."

#### Feature Engineering
- **FamilySize**: A new feature, `FamilySize`, is created by summing `SibSp` (siblings/spouses aboard) and `Parch` (parents/children aboard) plus 1 (for the passenger themselves). This feature represents the size of a passenger's family.

### 4. **Feature Selection**
The following features are selected for training the model:
- `Pclass`: Passenger class (1, 2, 3)
- `Sex`: Encoded gender (male=1, female=0)
- `Age`: Age of the passenger
- `Fare`: Fare paid for the ticket
- `FamilySize`: Size of the family
- `Embarked_Q`, `Embarked_S`: One-hot encoded embarkation ports (Q, S)

### 5. **Splitting the Data into Training and Testing Sets**
The dataset is split into training and testing sets using `train_test_split()`. The training set comprises 80% of the data, while the remaining 20% is used for testing the model. The split is done with a random seed of 42 to ensure reproducibility.

### 6. **Model Training**
We use a **Random Forest Classifier** for the classification task. The model is initialized with 100 trees (`n_estimators=100`) and trained on the training data using `rf_model.fit(X_train, y_train)`.

### 7. **Model Evaluation**
Once the model is trained, predictions are made on the test set using `rf_model.predict(X_test)`. The model's performance is evaluated using the following metrics:
- **Accuracy**: The proportion of correct predictions (i.e., the number of survivors and non-survivors predicted correctly).
- **Classification Report**: Provides additional metrics such as precision, recall, and F1-score for each class (survived or not survived).
- **Confusion Matrix**: Shows the count of true positives, true negatives, false positives, and false negatives.

These metrics are printed to evaluate how well the model performs in predicting the survival of passengers.

### 8. **Saving the Trained Model**
After training and evaluating the model, it is saved to a file using `joblib.dump()`. The model is saved as `'random_forest_titanic_model.pkl'` so that it can be reused later without retraining.

### 9. **Making Predictions with the Saved Model**
To demonstrate how the saved model can be used for future predictions, new passenger data is provided in a `pandas` DataFrame with the required features. The saved model is loaded using `joblib.load()` and used to predict the survival of the new passengers.

The following new passenger data is used for prediction:
```python
new_passengers = pd.DataFrame({
    'Pclass': [3, 1],
    'Sex': [1, 0],  # Male=1, Female=0
    'Age': [25, 38],
    'Fare': [7.25, 71.28],
    'FamilySize': [1, 3],
    'Embarked_Q': [0, 0],
    'Embarked_S': [1, 0]
})
```

The predictions for the new passengers are printed, where `1` indicates the passenger survived and `0` indicates the passenger did not survive.

### Example Prediction Output:
```python
Predictions for new passengers:
[0 1]
```
This means the first passenger did not survive, and the second passenger survived.

### 10. **Conclusion**
The Random Forest model successfully classifies Titanic passengers based on features such as age, sex, family size, fare, and embarkation port. The model achieves a high classification accuracy and can be used to predict survival status for new passengers. The trained model has been saved for future use and can be applied to make predictions on new data.

---
