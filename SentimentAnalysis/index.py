import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical, pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

# Load data
df = pd.read_csv('ecommerce_data.csv')
print(df)

# Define a transformer to convert the labels to categorical values

# Define a transformer to tokenize and pad the text data
tokenizer = Tokenizer(num_words=5000)

text_transformer = Pipeline(steps=[
    ('tokenize', tokenizer),
    ('pad',
     FunctionTransformer(lambda x: pad_sequences(x, maxlen=100, padding='post', truncating='post'), validate=False))
])

sentiment_transformer = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Define a column transformer to apply the transformers to the correct columns
preprocessor = ColumnTransformer(transformers=[
    ('label', sentiment_transformer, ['Sentiment']),
    ('text', text_transformer, ['Review'])
])

# Split data into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Fit the preprocessor on the training data and transform the training and testing data

train_data_preprocessed = preprocessor.fit_transform(train_data)
test_data_preprocessed = preprocessor.transform(test_data)

# Define the RNN model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=5000, output_dim=32, input_length=100),
    tf.keras.layers.SimpleRNN(units=64),
    tf.keras.layers.Dense(units=2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data_preprocessed['text'], train_data_preprocessed['label'], epochs=10,
          validation_data=(test_data_preprocessed['text'], test_data_preprocessed['label']))

new_texts = ['The customer service was great!', 'The product was not what I expected.',
             'I had a neutral experience with this company.']
new_data = pd.DataFrame({'Review': new_texts})
print("this is the new data : ", new_data)

# Transform the new data using the preprocessor
new_data_preprocessed = preprocessor.transform(new_data)

# Make predictions on the new data
predictions = model.predict(new_data_preprocessed['text'])

# Print the predictions
if __name__ == '__main__':
    for i, pred in enumerate(predictions):
        sentiment = 'Positive' if pred[0] > pred[1] else 'Negative'
        print(f'Text: {new_texts[i]} \nPredicted sentiment: {sentiment} ({pred})\n')

