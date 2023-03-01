import pandas as pd
from sklearn.preprocessing import OneHotEncoder

data = {
    "User_ID": [10, 20, 30, 40, 50, 60, 70],
    'Name': ['Tom', 'Joseph', 'Faith', 'John', 'Brenda', 'Ema-son', 'Ian', ],
    'Gender': ['Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Male', ],
    'Age': [20, 21, 19, 18, 21, 19, 18],
    "Salary": [2300, 23897, 10000, 19000, 23897, 23400, 22000],
    'purchased': ["No", "No", "yes", "No", "No", "No", "yes"]
}
dataFrame = pd.DataFrame(data)

if __name__ == '__main__':

    # drop redundant columns.in this code , the inplace named variable ensures that the original dataframe is modified,
    # rather than returning the modified version og the original one .

    cols = ["Gender", "Name"]
    dataFrame.drop(columns=cols, inplace=True)
    dataFrame.set_index('User_ID', inplace=True)

    dependent_Variables = dataFrame.iloc[:, :-1]
    target_Variables = dataFrame.iloc[:, -1]
    print(dependent_Variables)
    print(target_Variables)

    # note that in machine larning , the One Hot encoder will generate as many columns as the number  of features in the
    # Gender column(each  column with the attribute name as in the data above.)

    print("One Hot Encoding....")
    gender_encoder = OneHotEncoder()
    gender_encoded = gender_encoder.fit_transform(dataFrame[['Gender']])
    gender_encoded_df = pd.DataFrame(gender_encoded.toarray(), columns=['Female', 'Male'])
    X = pd.concat([dataFrame.drop(['Gender'], axis=1), gender_encoded_df], axis=1)
    print(X)
