import cv2
import numpy as np
from tensorflow import keras
from PIL import Image
import time

# Load model
print("Loading model...")
model = keras.models.load_model('models/best_model.h5')

# Traffic sign classes
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

# Start webcam
cap = cv2.VideoCapture(0)
print("\n Webcam started! Hold a traffic sign image in front of the camera.")
print("Press 'q' to quit, 'SPACE' to capture and predict\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Display instructions
    cv2.putText(frame, "Press SPACE to predict | Q to quit", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Traffic Sign Detection', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # Capture and predict on SPACE
    if key == ord(' '):
        # Get center crop
        h, w = frame.shape[:2]
        size = min(h, w)
        start_x = (w - size) // 2
        start_y = (h - size) // 2
        crop = frame[start_y:start_y+size, start_x:start_x+size]
        
        # Preprocess
        img = cv2.resize(crop, (32, 32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Predict
        predictions = model.predict(img, verbose=0)
        class_id = np.argmax(predictions[0])
        confidence = predictions[0][class_id] * 100
        
        print(f"\n{'='*60}")
        print(f"🚦 Predicted: {classes[class_id]}")
        print(f"�� Confidence: {confidence:.2f}%")
        print(f"{'='*60}\n")
        
        # Show result on frame for 3 seconds
        result_frame = frame.copy()
        cv2.putText(result_frame, f"{classes[class_id]}", 
                    (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result_frame, f"Confidence: {confidence:.1f}%", 
                    (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('Traffic Sign Detection', result_frame)
        cv2.waitKey(3000)
    
    # Quit on 'q'
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n👋 Webcam closed!")
