import numpy as np
from keras.layers import Input, Embedding, Concatenate, Dense, Flatten
from keras.models import Model
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split

from Recommendation_System.MovieRatingData import MovieRatingData

num_users = 1000
num_movies = 1200
num_genres = 10
num_characters = 256

data = MovieRatingData().data

# THE INPUT LAYERS
user_id_input = Input(
    shape=[1],
    name='user_id_input')

user_age_input = Input(
    shape=[1],
    name='user_age_input')

user_gender_input = Input(
    shape=[1],
    name='user_gender_input')

user_liked_movies_input = Input(
    shape=[5],
    name='user_liked_movies_input')

movie_id_input = Input(
    shape=[1],
    name='movie_id_input')

movie_genre_input = Input(
    shape=[1],
    name='movie_genre_input')

movie_characters_input = Input(
    shape=[4],
    name='movie_characters_input')

# EMBEDDING LAYERS
user_id_embedding = Embedding(
    input_dim=num_users,
    output_dim=10,
    name='user_id_embedding')(user_id_input)

user_age_embedding = Embedding(
    input_dim=100,
    output_dim=5,
    name='user_age_embedding')(user_age_input)

user_gender_embedding = Embedding(
    input_dim=2,
    output_dim=3,
    name='user_gender_embedding')(user_gender_input)

user_liked_movies_embedding = Embedding(
    input_dim=num_movies,
    output_dim=10,
    name='user_liked_movies_embedding')(user_liked_movies_input)

movie_id_embedding = Embedding(
    input_dim=num_movies,
    output_dim=10,
    name='movie_id_embedding')(movie_id_input)

movie_genre_embedding = Embedding(
    input_dim=num_genres,
    output_dim=3,
    name='movie_genre_embedding')(movie_genre_input)

movie_characters_embedding = Embedding(
    input_dim=num_characters,
    output_dim=5,
    name='movie_characters_embedding')(movie_characters_input)

# Concatenate embeddings

user_embedding = Concatenate(name='user_embedding')(
    [Flatten()(user_id_embedding), Flatten()(user_age_embedding), Flatten()(user_gender_embedding),
     Flatten()(user_liked_movies_embedding)])

movie_embedding = Concatenate(name='movie_embedding')(
    [Flatten()(movie_id_embedding), Flatten()(movie_genre_embedding), Flatten()(movie_characters_embedding)])

"""Compute dot product and flatten.Its not must that you compute the Dot product.addition of the two embeddings is 
sometimes the best option.Note that we are only doing  flattening once too."""

dot_product = Flatten()(Concatenate()([user_embedding, movie_embedding]))

# Define output layer: has one neuron and the activation is linear meaning that the output will be the weighted sum
# of the inputs .
output = Dense(units=1, activation='linear', name='output')(dot_product)

# Define model
model = Model(inputs=[user_id_input, user_age_input, user_gender_input, user_liked_movies_input, movie_id_input,
                      movie_genre_input, movie_characters_input], outputs=output)

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])


def predict_rating_for_one_user(user_id, user_age, user_gender, user_liked_movies, movie_id, movie_genre,
                                movie_characters):
    user_id = [[user_id]]
    user_age = [[user_age]]
    gender_input = [[user_gender]]
    liked_movies_input = [user_liked_movies]
    unrated_movie_id = [[movie_id]]

    movie_genre = [[movie_genre]]

    movie_characters = [movie_characters]

    prediction = model.predict(
        [user_id, user_age, gender_input, liked_movies_input, unrated_movie_id, movie_genre,
         movie_characters])[0][0]
    return prediction


def model_training():
    user_ids = [d['user']['id'] for d in data]
    movie_ids = [d['movie']['id'] for d in data]
    age_values = [d['user']['age'] for d in data]
    gender_values = [d['user']['gender'] for d in data]
    like_movie_ids = [d['user']['likeMovies'] for d in data]
    genre_values = [d['movie']['genre'] for d in data]
    characters_values = [d['movie']['characters'] for d in data]
    ratings = [d['rating'] for d in data]

    user_train, user_test, movie_train, movie_test, age_train, age_test, gender_train, gender_test, like_movie_train, like_movie_test, genre_train, genre_test, characters_train, characters_test, ratings_train, ratings_test = train_test_split(
        user_ids, movie_ids, age_values, gender_values, like_movie_ids, genre_values, characters_values, ratings,
        test_size=0.2, random_state=42)

    # Convert the data to numpy arrays for  efficiency in accessing and processing the data .
    user_train = np.array(user_train)
    user_test = np.array(user_test)
    movie_train = np.array(movie_train)
    movie_test = np.array(movie_test)
    age_train = np.array(age_train)
    age_test = np.array(age_test)
    gender_train = np.array(gender_train)

    gender_test = np.array(gender_test)
    padded_liked_movies = pad_sequences(like_movie_train, maxlen=5, padding='post', truncating='post')
    print(padded_liked_movies)

    like_movie_train = np.array(padded_liked_movies)
    like_movie_test = np.array(like_movie_test)
    genre_train = np.array(genre_train)
    genre_test = np.array(genre_test)
    characters_train = np.array(characters_train)
    characters_test = np.array(characters_test)
    ratings_train = np.array(ratings_train)
    ratings_test = np.array(ratings_test)

    model.fit([user_train, movie_train, age_train, gender_train, like_movie_train, genre_train, characters_train],
              ratings_train, epochs=10, batch_size=32, validation_data=(
        [user_test, movie_test, age_test, gender_test, like_movie_test, genre_test, characters_test], ratings_test))


if __name__ == '__main__':
    model_training()
    oneUserPrediction = predict_rating_for_one_user(1200, 23, "male", [4000, 474646, 4747], 1230, "education",
                                                    ["Steve james"])
    print("Predicted rating for movie 1230 is : {}".format(oneUserPrediction))
