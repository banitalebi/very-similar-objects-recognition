import tensorflow as tf
from tensorflow.keras import layers, models, optimizers # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore

# Hyperparameters
epochs = 5
batch_size = 64
height = 224
width = 224

train_path = "dataset/train"
test_path = "dataset/test"

# Data augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# Load training data
train_generator = datagen.flow_from_directory(
    train_path,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Load validation data
validation_generator = datagen.flow_from_directory(
    train_path,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Prepare test data
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='categorical'
)

num_classes = train_generator.num_classes

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(height, width, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Create a ModelCheckpoint callback to save the best model
checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    mode='min'
)

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    callbacks=[checkpoint]
)

# Load the best model
best_model = tf.keras.models.load_model('best_model.keras')

# Evaluate on test set using the best model
test_loss, test_accuracy = best_model.evaluate(test_generator)
print(f"Test accuracy: {test_accuracy:.4f}")
