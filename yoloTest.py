from ultralytics import YOLO

if __name__ == "__main__":
    trained_model_path = "runs/detect/traffic_signs_yolov89/weights/best.pt"
    model = YOLO(trained_model_path)

    results = model.predict(
        source="test/images",       # testing images
        show=True,                  # display images while predicting
        save=True,                  # save annotated images
        project="runs/predict",     # folder to save predictions
        name="annotated_test"       # subfolder name
    )

    for r in results:
        print("Boxes:", r.boxes.xyxy)
        print("Confidences:", r.boxes.conf)
        print("Class IDs:", r.boxes.cls)
