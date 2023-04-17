from keras.layers import Input, Embedding, Flatten, Dot, Concatenate
from keras.models import Model


data = [{'user': {'id': 1001, 'age': 25, 'likeMovies': [18, 25, 357, 395000], 'gender': 'male'},
         'movie': {'id': 8989, 'genre': 'education', 'characters': ['james pater', 'David', 'Saul', 'Mike']},
         'rating': 3},

        {'user': {'id': 1002, 'age': 30, 'likeMovies': [25, 35, 67, 123, 765], 'gender': 'female'},
         'movie': {'id': 2345, 'genre': 'action', 'characters': ['Tom', 'John', 'Mary', 'Samantha']},
         'rating': 4},

        {'user': {'id': 1003, 'age': 20, 'likeMovies': [18, 35, 67, 123, 8989], 'gender': 'male'},
         'movie': {'id': 395000, 'genre': 'comedy', 'characters': ['Steve', 'Linda', 'Joe', 'Rachel']},
         'rating': 5},
        ]

num_users = len(set([d['user']['id'] for d in data]))
num_movies = len(set([d['movie']['id'] for d in data]))

user_id_input = Input(shape=[1], name='user_id_input')
movie_id_input = Input(shape=[1], name='movie_id_input')

user_embedding = Embedding(input_dim=num_users, output_dim=10, name='user_embedding')(user_id_input)
movie_embedding = Embedding(input_dim=num_movies, output_dim=10, name='movie_embedding')(movie_id_input)

user_flattened = Flatten()(user_embedding)
movie_flattened = Flatten()(movie_embedding)

concatenated = Concatenate()([user_flattened, movie_flattened])

dot_product = Dot(axes=1)([user_flattened, movie_flattened])

model = Model(inputs=[user_id_input, movie_id_input], outputs=dot_product)

model.compile(loss='mse', optimizer='adam')

users = [d['user']['id'] for d in data]
movies = [d['movie']['id'] for d in data]
ratings = [d['rating'] for d in data]

# fit the model on the input data
model.fit([users, movies], ratings, epochs=10, batch_size=32)


"""TRAIN THE MODE: we pass (users and ratings as inputs) and the ground truth(Ratings) to the model; for any dot 
product between  the  users and the movie ratings , the output is compared to the ground truth and any errors back 
propagated.This means that the ratings will be compared to  the outputs=flatten in the model."""

