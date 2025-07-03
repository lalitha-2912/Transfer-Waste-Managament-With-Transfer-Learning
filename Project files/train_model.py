from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Paths
train_dir = os.path.join('Dataset', 'train')
test_dir = os.path.join('Dataset', 'test')

print("Train dir:", train_dir)
print("Test dir:", test_dir)
print("Train classes",os.listdir(train_dir))
print("Test classes:", os.listdir(test_dir))
for c in os.listdir(train_dir):
    print(f"Train/{c} has {len(os.listdir(os.path.join(train_dir, c)))} images")
for c in os.listdir(test_dir):
    print(f"Test/{c} has {len(os.listdir(os.path.join(test_dir, c)))} images")

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Model
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
vgg.trainable = False

model = Sequential([
    vgg,
    Flatten(),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

# Save
model.save('healthy_vs_rotten.h5')
print('Model saved as healthy_vs_rotten.h5') 