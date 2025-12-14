from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("runs/detect/traffic_signs_yolov89/weights/best.pt")
    
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