from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

data = {
    'Name': ['Tom', 'Joseph', 'Faith', 'John', 'Brenda', 'Ema-son', 'Ian', ],
    'Gender': ['Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Male', ],
    'Age': [20, 21, 19, 18, 21, 19, 18],
    "Salary": [2300, 23897, 10000, 19000, 23897, 23400, 22000],
    'purchased': ["No", "No", "yes", "No", "No", "No", "yes"]
}
df = pd.DataFrame(data)

"""note that most functions in AI accept a 2D array , that is , in  form  of a  column .again , we set the random 
state so that for the same input data, then we have the same output .y_Data in the case below is also a dataFrame.
The feature names allows us to see how the rues are used eg salary<= 10000 is a rule , 
 we also set the className for each node of the tree , so that we are able  to see the  className of each node."""
if __name__ == '__main__':
    newData = df.drop(columns=["Name"])

    y_Data = newData["purchased"]
    x_Data = newData.drop(columns=["purchased"])
    x_train, x_test, y_train, y_test = train_test_split(x_Data, y_Data, test_size=.1, random_state=123, )

    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    tree.export_graphviz(model, out_file="purchase_predictor.dot", feature_names=["Age", "Salary"], class_names=y_Data.unique(), label="all", rounded=True, filled=True)
    predictions = model.predict(x_test)
    print(predictions)

    print(accuracy_score(y_test, predictions))

