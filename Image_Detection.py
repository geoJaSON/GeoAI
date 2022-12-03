from imageai.Detection import ObjectDetection
import os
detector = ObjectDetection()
#detector.setModelTypeAsYOLOv3()

detector.setModelTypeAsRetinaNet()
detector.setModelPath(r"C:\Users\jason\Downloads\resnet50_coco_best_v2.0.1.h5")
#detector.setModelPath(r"C:\Users\jason\Downloads\yolo.h5")
detector.loadModel()

detections = detector.detectObjectsFromImage(input_image=r"C:\Users\jason\github\GIS_Tasks\Geo_AI\street.jpg", output_image_path=r"C:\Users\jason\github\GIS_Tasks\Geo_AI\"street_detected.jpg")

for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("--------------------------------")
import sys
sys.executable
import cv2
