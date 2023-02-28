import pandas as pd
from sklearn.preprocessing import OneHotEncoder

data = {
    'Name': ['Tom', 'Joseph', 'Faith', 'John', 'Brenda', 'Ema-son', 'Ian', ],
    'Gender': ['Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Male', ],
    'Age': [20, 21, 19, 18, 21, 19, 18],
    "Salary": [2300, 23897, 10000, 19000, 23897, 23400, 22000],
    'purchased': ["No", "No", "yes", "No", "No", "No", "yes"]
}
dataFrame = pd.DataFrame(data)

if __name__ == '__main__':
    print("Pne hot encoding....")
    gender_encoder = OneHotEncoder()
    gender_encoded = gender_encoder.fit_transform(dataFrame[['Gender']])
    gender_encoded_df = pd.DataFrame(gender_encoded.toarray(), columns=['Female', 'Male'])
    X = pd.concat([dataFrame.drop(['Gender'], axis=1), gender_encoded_df], axis=1)
    print(X)

