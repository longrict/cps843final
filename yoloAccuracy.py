from ultralytics import YOLO

if __name__ == "__main__":
    modelNames = [
        "yolov8m_baseline",
        "yolov8m_BS8",
        "yolov8m_IS320",
        "yolov8m_IS512",

        "yolov8n_baseline",
        "yolov8n_BS8",
        "yolov8n_IS320",
        "yolov8n_IS512",

        "yolov8s_baseline",
        "yolov8s_BS8",
        "yolov8s_IS320",
        "yolov8s_IS512",

        "yolov11s_baseline",
        "yolov11s_BS8",
        "yolov11s_IS320",
        "yolov11s_IS512"
    ]

    for name in modelNames:
        # get accuracy for each model
        model = YOLO("runs/detect/"+ name +"/weights/best.pt")
        
        print("Running accuracy test...\n")
        
        results = model.val(data="trafficsigns.yaml", split="val")
        
        print("OVERALL ACCURACY")
        print("Precision: " + str(round(results.box.mp * 100, 1)) + "% of detections were correct")
        print("Recall: " + str(round(results.box.mr * 100, 1)) + "% of signs were found")
        print("mAP@0.5: " + str(round(results.box.map50 * 100, 1)) + "% overall accuracy\n")


        print("PER SIGN ACCURACY")
        
        names = model.names
        for i, ap in enumerate(results.box.ap50):
            print(names[i] + ": " + str(round(ap * 100, 1)) + "%")