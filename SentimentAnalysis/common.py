import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical, pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

import string

# Load data
df = pd.read_csv('ecommerce_data.csv')


def labelEncodeSentiments(data_frame):
    label_map = {' negative': 0, ' positive': 1, ' neutral': 2}
    data_frame['Sentiment'] = data_frame['Sentiment'].map(label_map)
    return data_frame


def preprocessText(text):
    # Remove multiple white spaces
    text = ' '.join(text.split())

    # Remove single characters
    text = ' '.join([word for word in text.split() if len(word) > 1])

    # Remove punctuation marks
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = ''.join([i for i in text if not i.isdigit()])

    return text


dataFrame = labelEncodeSentiments(df)

tokenizer = Tokenizer(num_words=5000)
X_Data = dataFrame['Review']
labels = dataFrame['Sentiment']

tokenizer.fit_on_texts(X_Data)
sequences = tokenizer.texts_to_sequences(X_Data)

max_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post")

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

model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=64, input_length=max_length))
model.add(LSTM(64))
model.add(Dense(1, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(padded_sequences, labels, batch_size=1, epochs=10, validation_split=0.2)

new_texts = ['The customer service was great!', 'The product was not what I expected.',
             'I had a neutral experience with this company.', 'I cant stand the new update']

new_sequences = tokenizer.texts_to_sequences(pd.Series(new_texts))
new_padded_sequences = pad_sequences(new_sequences, maxlen=max_length, padding="post")

print(new_padded_sequences)

# Transform the new data using the preprocessor


# Make predictions on the new data
predictions = model.predict(new_padded_sequences)

# Print the predictions
if __name__ == '__main__':
    print("Predictions are : ", predictions)
    for i, pred in enumerate(predictions):
        print(pred)
