from keras.layers import Input, Embedding, Concatenate, Dense, Flatten
from keras.models import Model

num_users = 1000
num_movies = 1200
num_genres = 10
num_characters = 256

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
    shape=[5],
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
    [user_id_embedding, user_age_embedding, user_gender_embedding, user_liked_movies_embedding])

movie_embedding = Concatenate(name='movie_embedding')(
    [movie_id_embedding, movie_genre_embedding, movie_characters_embedding])

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


if __name__ == '__main__':
    oneUserPrediction = predict_rating_for_one_user(1200, 23, "male", [4000, 474646, 4747], 1230, "education",
                                                    ["Steve james"])
    print("Predicted rating for movie 1230 is : {}".format(oneUserPrediction))
