from ultralytics import YOLO

if __name__ == "__main__":
    # Load YOLOv8 model (small/medium/large)
    model = YOLO("yolov8n.pt")  # "n" = nano (fast, small), or "s" = small

    # Train
    model.train(
        data="trafficsigns.yaml",
        epochs=50,
        imgsz=416,
        batch=16,
        name="traffic_signs_yolov8",
        device=0  # GPU 0
    )


