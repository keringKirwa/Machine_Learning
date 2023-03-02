import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
data = {
    "User_ID": [10, 20, 30, 40, 50, 60, 70],
    'Name': ['Tom', 'Joseph', 'Faith', 'John', 'Brenda', 'Ema-son', 'Ian', ],
    'Gender': ['Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Male', ],
    'Age': [20, 21, 19, 18, 21, 19, 18],
    "Salary": [2300, 23897, 10000, 19000, 23897, 23400, 22000],
    'purchased': ["No", "No", "yes", "No", "No", "No", "yes"]
}
data = pd.DataFrame(data)
print(data)

# Split the data into features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Encode the categorical variable
gender_encoder = OneHotEncoder()
gender_encoded = gender_encoder.fit_transform(X[['Gender']])
gender_encoded_df = pd.DataFrame(gender_encoded.toarray(), columns=['Female', 'Male'])
print(gender_encoded_df)
X = pd.concat([X.drop(['Gender'], axis=1), gender_encoded_df], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

""" Construct the decision tree : can also use entropy as a split criterion , entropy specifies 
the function that uses information gain for splitting.
"""

clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

if __name__ == '__main__':
    print(clf.random_state)
