import numpy as np
from keras.layers import Input, Embedding, Dot, Flatten
from keras.models import Model

# define number of users, movies, and embedding dimensions
num_users = 100
num_movies = 30
embedding_size = 48

# define inputs
user_id_input = Input(shape=(1,))
movie_id_input = Input(shape=(1,))

# define user and movie embeddings
user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size)(user_id_input)
movie_embedding = Embedding(input_dim=num_movies, output_dim=embedding_size)(movie_id_input)

# dot product of user and movie embeddings
dot_product = Dot(axes=-1)([user_embedding, movie_embedding])

# flatten the dot product tensor
flatten = Flatten()(dot_product)

# define the model
model = Model(inputs=[user_id_input, movie_id_input], outputs=flatten)

# compile the model
model.compile(loss='mse', optimizer='adam')

# generate some dummy data for training
users = np.random.randint(num_users, size=1000)
movies = np.random.randint(num_movies, size=1000)
ratings = np.random.randint(1, 6, size=1000)

# train the model
model.fit([users, movies], ratings, epochs=10, batch_size=32)

# make a prediction for a user-movie pair
user_id = np.array([10])
movie_id = np.array([5])
rating = model.predict([user_id, movie_id])

print("The predicted rating for user {} and movie {} is {}".format(user_id[0], movie_id[0], rating[0]))
