import numpy as np
from keras.layers import Input, Embedding, Flatten, Dot, Concatenate, Dense
from keras.models import Model
from keras.optimizers import Adam

embedding_size = 48
num_users = 100
num_movies = 30

user_input = Input(shape=(1,))
movie_input = Input(shape=(1,))
age_input = Input(shape=(1,))
gender_input = Input(shape=(1,))

movie_embedding = Embedding(input_dim=num_movies, output_dim=48)(movie_input)

# users
user_embedding = Embedding(input_dim=num_users, output_dim=16)(user_input)
age_embedding = Embedding(input_dim=100, output_dim=16)(age_input)
gender_embedding = Embedding(input_dim=2, output_dim=16)(gender_input)

user_embedding = Flatten()(user_embedding)
movie_embedding = Flatten()(movie_embedding)
age_embedding = Flatten()(age_embedding)
gender_embedding = Flatten()(gender_embedding)

user_embedding = Concatenate()([user_embedding, age_embedding, gender_embedding])
movie_embedding = Concatenate()([movie_embedding])

rating = Dot(axes=-1)([user_embedding, movie_embedding])

model = Model(inputs=[user_input, movie_input, age_input, gender_input], outputs=rating)

model.compile(loss='mse', optimizer=Adam(lr=0.001))

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

"""
Note the following arrays are of the shape (4, 1) , meaning , they have 4 rows and 1 n column"""
test_users = np.array([[24], [30], [45], [50]])
test_movies = np.array([[24], [10], [29], [10]])
ratings = model.predict(
    [test_users, test_movies, np.array([[30], [40], [25], [18]]), np.array([[0], [1], [1], [0]])])
print(ratings)
if __name__ == '__main__':
    for i in range(4):
        print("User {} is predicted to rate movie {} as {} out of 5".format(
            test_users[i][0], test_movies[i][0], ratings[i]))

# Print the predicted ratings
