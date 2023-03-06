import numpy as np
from keras.layers import Input, Embedding, Flatten, Dot, Concatenate, Dense
from keras.models import Model
from keras.optimizers import Adam
"""
Note that the Convolutional neural network (CNN) is a sequential  model , meaning : Each layer takes the output of the 
previous layer as input and produces its own output, which is fed as input to the next layer. This sequential 
arrangement allows the network to gradually learn and extract more complex features from the input data.
"""

num_users = 50
num_movies = 100

embedding_size = 16

# Create input layers for user and movie IDs.The input layer of the neaural network accepts an array/tensor with one
# dimension , and one element at a time eg [10],then  [20] ...

user_input = Input(shape=(1,))
movie_input = Input(shape=(1,))

age_input = Input(shape=(1,))
gender_input = Input(shape=(1,))

user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size)(user_input)
movie_embedding = Embedding(input_dim=num_movies, output_dim=embedding_size)(movie_input)

age_embedding = Embedding(input_dim=100, output_dim=embedding_size)(age_input)
gender_embedding = Embedding(input_dim=2, output_dim=embedding_size)(gender_input)


user_embedding = Flatten()(user_embedding)
movie_embedding = Flatten()(movie_embedding)

age_embedding = Flatten()(age_embedding)
gender_embedding = Flatten()(gender_embedding)

user_embedding = Concatenate()([user_embedding, age_embedding, gender_embedding])
movie_embedding = Concatenate()([movie_embedding])

rating = Dot(axes=-1)([user_embedding, movie_embedding])

# Combine all inputs and outputs into a single model
model = Model(inputs=[user_input, movie_input, age_input, gender_input], outputs=rating)

# Compile the model with a mean squared error loss function and an Adam optimizer
model.compile(loss='mse', optimizer=Adam(lr=0.001))

"""Generate some example data"""
user_ids = np.random.randint(num_users, size=30)
movie_ids = np.random.randint(num_movies, size=30)
ages = np.random.randint(18, 60, size=30)
genders = np.random.randint(2, size=30)
print(genders)
ratings = np.random.randint(1, 6, size=30)

# Fit the model on the example data
model.fit([user_ids, movie_ids, ages, genders], ratings, epochs=10, batch_size=10, verbose=2)
print(model.predict())

if __name__ == '__main__':

    test_users = [[24, 30, 45, 50]]
    test_movies = [[90, 88, 67, 10]]
    ratings = model.predict([test_users, test_movies, np.zeros_like(test_users), np.zeros_like(test_users)])
    print(ratings)

    for i in range(len(test_users)):
        print("User {} is predicted to rate movie {} as {:.2f} out of 5".format(
            test_users[0][i], test_movies[0][i], ratings[i]))
