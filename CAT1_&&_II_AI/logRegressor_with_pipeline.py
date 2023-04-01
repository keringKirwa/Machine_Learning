from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import pandas as pd

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', None)

df = pd.read_excel("/home/arapkering/Downloads/churn_prediction.xlsx", sheet_name='E Comm')


def predict_churn(customer_details, model):
    new_instance_df = pd.DataFrame([customer_details])
    return model.predict(new_instance_df)[0]


if __name__ == '__main__':
    dataFrame = df.drop("CustomerID", axis=1)
    columns_of_interest = ['NumberOfDeviceRegistered', 'PreferedOrderCat', 'Tenure', 'Gender', 'OrderCount', 'Churn']

    data_frame_of_interest = df[columns_of_interest].dropna(subset=['OrderCount', 'Tenure'])

    numerical_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessing = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, ['NumberOfDeviceRegistered', 'Tenure', 'OrderCount']),
            ('cat', categorical_transformer, ['PreferedOrderCat', 'Gender'])
        ]
    )

    lr_pipe = Pipeline([
        ('preprocessing', preprocessing),
        ('logistic_regression', LogisticRegression())
    ])

    X = data_frame_of_interest.drop(['Churn'], axis=1)
    y = data_frame_of_interest['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr_pipe.fit(X_train, y_train)

    y_pred = lr_pipe.predict(X_test)

    # Model Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)

    instance = {
        "NumberOfDeviceRegistered": 2,
        "PreferedOrderCat": "Mobiles",
        "Tenure": 10,
        "Gender": "Female",
        "OrderCount": 3
    }

    prediction_for_one_customer = predict_churn(instance, lr_pipe)
    print("Prediction for one Customer: {}".format(prediction_for_one_customer))
