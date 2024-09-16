import tensorflow as tf
from tensorflow.keras import layers, models, optimizers # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# Hyperparameters
epochs = 9
batch_size = 24
learning_rate = 0.0001
height = 224
width = 224

train_path = "dataset/train"
validation_path = "dataset/validation"
test_path = "dataset/test"

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=10,
    brightness_range=[0.9, 1.1],
    zoom_range=0.05,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_datagen = ImageDataGenerator(rescale=1.0/255)

validation_generator = validation_datagen.flow_from_directory(
    validation_path,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='categorical',
)


test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Define the model
model = models.Sequential()
model.add(tf.keras.applications.MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(height, width, 3)))
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(len(train_generator.class_indices), activation='softmax'))

# Compile the model
model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
# Train the model
model.fit(train_generator,
          validation_data=validation_generator,
          epochs=epochs)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test results: Test loss: {test_loss:.4f} | Test accuracy: {test_accuracy:.2f}%')
