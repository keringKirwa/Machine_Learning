from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np

import pandas as pd

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', None)

dataFrame = pd.read_csv('/home/arapkering/Desktop/winequality-red.csv')

if __name__ == '__main__':
    dataFrame['is_good'] = np.where(dataFrame['quality'] >= 6, 1, 0)
    col_names = list(dataFrame.columns)

    col_names.remove('is_good')
    col_names.remove("quality")

    numerical_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessing = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, col_names),
        ]
    )

    lr_model = Pipeline([
        ('preprocessing', preprocessing),
        ('logistic_regression', LogisticRegression())
    ])

    X = dataFrame.drop(['quality', 'is_good'], axis=1)
    Y = dataFrame['is_good']

    print("This is Y data : ", Y)
    print(type(Y))

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    lr_model.fit(X_train, y_train)

    y_predicted = lr_model.predict(X_test)

    # Model Evaluation
    accuracy = accuracy_score(y_test, y_predicted)

    print('Accuracy:', accuracy)
    print(classification_report(y_test, y_predicted))
