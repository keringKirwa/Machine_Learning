import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("customer_data.csv")
print(data.drop())

# Split the data into features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Encode the categorical variable
gender_encoder = OneHotEncoder()
gender_encoded = gender_encoder.fit_transform(X[['Gender']])
gender_encoded_df = pd.DataFrame(gender_encoded.toarray(), columns=['Female', 'Male'])
X = pd.concat([X.drop(['Gender'], axis=1), gender_encoded_df], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pd.concat([X.drop()], axis=1)


# Construct the decision tree
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

if __name__ == '__main__':
    print(clf.random_state)
