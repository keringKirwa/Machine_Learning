import tensorflow as tf

# Define the image input shape
image_input_shape = (150, 150, 3)

# Define the image model
image_input = tf.keras.layers.Input(shape=image_input_shape)
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(image_input)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Flatten()(x)
image_output = tf.keras.layers.Dense(128, activation='relu')(x)

# Define the metadata input shape
metadata_input_shape = (10,)

# Define the metadata model
metadata_input = tf.keras.layers.Input(shape=metadata_input_shape)
metadata_output = tf.keras.layers.Dense(128, activation='relu')(metadata_input)

# Define the text input shape
text_input_shape = (100,)

# Define the text model
text_input = tf.keras.layers.Input(shape=text_input_shape)
text_output = tf.keras.layers.Dense(128, activation='relu')(text_input)

# Concatenate the image, metadata, and text outputs
concatenated = tf.keras.layers.Concatenate()([image_output, metadata_output, text_output])

# Add a final classification layer
output = tf.keras.layers.Dense(3, activation='softmax')(concatenated)

# Define the full model
model = tf.keras.models.Model(inputs=[image_input, metadata_input, text_input], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()