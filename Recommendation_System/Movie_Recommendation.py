import numpy as np
from keras.layers import Input, Embedding, Flatten, Dot, Concatenate, Dense
from keras.models import Model
from keras.optimizers import Adam

embedding_size = 48
num_users = 100
num_movies = 30

# INPUTS

user_id_input = Input(shape=(1,))
age_input = Input(shape=(1,))
gender_input = Input(shape=(1,))

rated_movie_input = Input(shape=(1,))
other_movies_input = Input(shape=(10,), name='other_liked_movies')

"""NOTE: if a user has rated less than 10 movies, the input sequence will be padded with a special token to make it of 
length 10"""

# EMBEDDINGS
user_id_embedding = Embedding(input_dim=num_users, output_dim=16)(user_id_input)
age_embedding = Embedding(input_dim=100, output_dim=16)(age_input)
gender_embedding = Embedding(input_dim=2, output_dim=16)(gender_input)

rated_movie_embedding = Embedding(input_dim=num_movies, output_dim=48)(rated_movie_input)
# input_dim=len(movie_data['movie_id'].unique()).We are expecting an array of length 10 , hence input length is set
# to 10
other_movies_embedding = Embedding(input_dim=num_movies, output_dim=10, input_length=10)(other_movies_input
                                                                                         )

# FLATTENING LAYERS
user_id_embedding = Flatten()(user_id_embedding)
age_embedding = Flatten()(age_embedding)
gender_embedding = Flatten()(gender_embedding)

rated_movie_embedding = Flatten()(rated_movie_embedding)
other_movies_embedding = Flatten()(other_movies_embedding)

combined_embedding = Concatenate()([user_id_embedding, age_embedding, gender_embedding, rated_movie_embedding, other_movies_embedding])

"""Note that in a recommendation system , we are only working  with the tensors/arrays.This means that we cant pass 
integers , such as age=25 , instead , we pass it as a tensor such as age=np.array([25])"""

model = Model(inputs=[user_id_input, rated_movie_input, age_input, gender_input], outputs=rating)

model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

user_ids = np.random.randint(num_users, size=30)
movie_ids = np.random.randint(num_movies, size=30)
ages = np.random.randint(18, 60, size=30)
genders = np.random.randint(2, size=30)
ratings = np.random.randint(1, 6, size=30)

print("user_ids", user_ids.shape)
print(movie_ids.shape)
print(ages.shape)
print(genders.shape)

model.fit([user_ids, movie_ids, ages, genders], ratings, epochs=10, batch_size=10, verbose=2)
my_array = np.array([30, 40, 25, 18])

test_users = np.array([[24], [30], [45], [50]])
test_movies = np.array([[24], [10], [29], [10]])
ratings = model.predict(
    [test_users, test_movies, np.array([[30], [40], [25], [18]]), np.array([[1], [1], [1], [0]])])
print(ratings)


def prediction_for_one_user(user_id, movie_ids, user_age, user_gender):
    user_id = np.array([[user_id]])
    user_age = np.array([[user_age]])
    user_gender = np.array([[user_gender]])
    movie_id = np.array(movie_ids)
    movie_id = movie_id.reshape(-1, 1)
    one_user_rating = model.predict([user_id, np.array([[movie_id[0]]]), user_age, user_gender])
    print("rating  for one user :", one_user_rating)


if __name__ == '__main__':
    for i in range(4):
        print("User {} is predicted to rate movie {} as {} out of 5".format(
            test_users[i][0], test_movies[i][0], ratings[i]))
    prediction_for_one_user(87, [15, 12, 1], 13, 1)

# Print the predicted ratings
