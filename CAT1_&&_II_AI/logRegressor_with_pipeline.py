from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import pandas as pd

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', None)

df = pd.read_excel("/home/arapkering/Downloads/churn_prediction.xlsx", sheet_name='E Comm')

"""Note  (1)The  model  will be fed with  both the x and y data , which  too  have to be processed.(2) An transformer 
is  used in modifying hte data. (3) an estimator is used in learning from the data.Pipelines chain together processes 
that are applied t the entire dataset.In the case when we want  to apply a transformer to only one or some  column(s) 
, then we use a ColumnTransformer().Again note , we dont want to  scale  the  one hot encoded values ,  therefore the 
StandardScaler will and MUST always  come before  the OneHotEncoder. Every transformer must have a name  eg (scaler, 
onehot) in the  cases below and so on .they will  be used to refer to the transformers later in the program."""


def predict_churn(customer_details, model):
    new_instance_df = pd.DataFrame([customer_details])
    print(new_instance_df.shape)
    print(new_instance_df)
    # if new_instance_df.shape[0] == 1:
    #     new_instance_df.reshape(1, -1)
    # elif new_instance_df.shape[1] == 1:
    #     # new_instance_df.reshape(-1, 1)
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

    print(X_test.shape)

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
    lr_pipe.named_steps["logistic_regression"].predict(instance)
    print("sth very cool here :: ", lr_pipe.named_steps["logistic_regression"].predict(instance))
    print("Prediction for one Customer: {}".format(prediction_for_one_customer))
