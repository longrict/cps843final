import numpy as np
from PIL import Image
import os
from tensorflow import keras

print("Loading model...")
model = keras.models.load_model('models/best_model.h5')

print("Loading test data...")

# classes linked with folder name in 'data/Test'
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

# Load test data
X_test = []
y_test = []

for class_num in range(43):
    path = os.path.join('data/Test', str(class_num))
    if not os.path.exists(path):
        continue
        
    for img_file in os.listdir(path):
        try:
            img_path = os.path.join(path, img_file)
            image = Image.open(img_path)
            image = image.resize((32, 32))
            image = np.array(image)
            
            # Ensure RGB
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                image = image[:, :, :3]
            
            X_test.append(image)
            y_test.append(class_num)
        except:
            pass

X_test = np.array(X_test, dtype=np.float32) / 255.0
y_test = np.array(y_test)

print(f"Test samples: {len(X_test)}")
print(f"Test data shape: {X_test.shape}")

# Predict
predictions = model.predict(X_test, batch_size=32)
predicted_classes = np.argmax(predictions, axis=1)

# Calculate accuracy
accuracy = np.mean(predicted_classes == y_test)
print(f"\n Test Accuracy: {accuracy * 100:.2f}%")

# Show some predictions
print("\nSample Predictions:")
for i in range(10):
    print(f"True: {classes[y_test[i]]} | Predicted: {classes[predicted_classes[i]]} | {'✓' if y_test[i] == predicted_classes[i] else '✗'}")