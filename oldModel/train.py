import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
import platform
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# GPU Configuration - Works on both Mac and Windows
print("="*60)
print("🖥️  System Configuration")
print("="*60)
print(f"Operating System: {platform.system()} {platform.release()}")
print(f"Python Version: {platform.python_version()}")
print(f"TensorFlow Version: {tf.__version__}")

# Check for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth for GPU (prevents OOM errors)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Available: {len(gpus)} GPU(s) detected")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
    except RuntimeError as e:
        print(f"⚠️  GPU Configuration Error: {e}")
else:
    print("⚠️  No GPU detected - using CPU")
    print("   Tip: Training will be slower on CPU")
print("="*60)
print()

print("Loading dataset...")

# Traffic sign class names
classes = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection', 12: 'Priority road', 13: 'Yield',
    14: 'Stop', 15: 'No vehicles', 16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry', 18: 'General caution', 19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right', 21: 'Double curve', 22: 'Bumpy road',
    23: 'Slippery road', 24: 'Road narrows on the right', 25: 'Road work',
    26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End of all speed and passing limits', 33: 'Turn right ahead',
    34: 'Turn left ahead', 35: 'Ahead only', 36: 'Go straight or right',
    37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left',
    40: 'Roundabout mandatory', 41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

# Load training data - Cross-platform path handling
def load_data(data_dir):
    images = []
    labels = []
    
    for class_num in range(43):
        path = os.path.join(data_dir, str(class_num))
        if not os.path.exists(path):
            continue
            
        print(f"Loading class {class_num}: {classes[class_num]}")
        
        for img_file in os.listdir(path):
            try:
                img_path = os.path.join(path, img_file)
                image = Image.open(img_path)
                image = image.resize((32, 32))
                image = np.array(image)
                
                # Ensure RGB format (3 channels)
                if len(image.shape) == 2:  # Grayscale
                    image = np.stack([image] * 3, axis=-1)
                elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
                    image = image[:, :, :3]
                    
                images.append(image)
                labels.append(class_num)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels)
    
    return images, labels

# Load data - Works on both Windows and Mac
X_train, y_train = load_data(os.path.join('data', 'Train'))
X_test, y_test = load_data(os.path.join('data', 'Test'))

print(f"\nDataset loaded!")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Convert labels to categorical
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

# Split training data into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train)}")
print(f"Validation set: {len(X_val)}")
print(f"Test set: {len(X_test)}")

# Build CNN Model
print("\nBuilding model...")

model = Sequential([
    # First Convolutional Block
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Second Convolutional Block
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Third Convolutional Block
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Flatten and Dense Layers
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(43, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    horizontal_flip=False
)

datagen.fit(X_train)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Callbacks
checkpoint = ModelCheckpoint(
    os.path.join('models', 'best_model.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.00001,
    verbose=1
)

# Train the model
print("\nStarting training...")

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=30,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

# Evaluate on test set
print("\nEvaluating on test set...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Save final model
model.save(os.path.join('models', 'traffic_sign_model.h5'))
print("\nModel saved as 'models/traffic_sign_model.h5'")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join('models', 'training_history.png'))
print("Training history plot saved as 'models/training_history.png'")
plt.show()