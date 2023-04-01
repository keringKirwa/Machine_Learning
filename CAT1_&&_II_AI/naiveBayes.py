import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', None)

df = pd.read_excel("/home/arapkering/Downloads/churn_prediction.xlsx", sheet_name='E Comm')


def predict_churn(customer_details, model):
    new_instance_df = pd.DataFrame([customer_details])
    df_encoded = label_encoder(new_instance_df, ["PreferedOrderCat", "Gender"])

    return model.predict(df_encoded)[0]


def label_encoder(main_data_frame, categorical_cols):
    df_to_encode = main_data_frame[categorical_cols]
    naive_label_encoder = LabelEncoder()
    df_encoded = df_to_encode.apply(naive_label_encoder.fit_transform)
    return pd.concat([main_data_frame.drop(categorical_cols, axis=1), df_encoded], axis=1)


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


if __name__ == '__main__':
    dataFrame = df.drop("CustomerID", axis=1)
    dataFrame.describe(include='O')
    columns_of_interest = ['NumberOfDeviceRegistered', 'PreferedOrderCat', 'Tenure', 'Gender', 'OrderCount', 'Churn']
    data_frame_of_interest = df[columns_of_interest]

    numericalDF = label_encoder(data_frame_of_interest, ["PreferedOrderCat", "Gender"])

    mean_tenure = numericalDF["Tenure"].mean()

    numericalDF = numericalDF[numericalDF['OrderCount'].notna()]
    numericalDF = numericalDF[numericalDF['Tenure'].notna()]

    classes = numericalDF["Churn"]
    X_main_data = numericalDF.drop(["Churn"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X_main_data, classes, test_size=0.2, random_state=42)

    print("Testing if data  has any Nan value :\n", X_test.isna().any())

    # Training the Naive Bayes classifier

    nb = GaussianNB()
    nb.fit(X_train, y_train)

    y_pred = nb.predict(X_test)

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

    prediction_for_one_customer = predict_churn(instance, nb)
    print("prediction for one Customer :: {}".format(prediction_for_one_customer))
