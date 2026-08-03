import sys
import pathlib
from datetime import datetime

sys.modules["pathlib._local"] = pathlib

from ultralytics import YOLO
import cv2
import numpy as np


class YOLODetector:
    def __init__(self, model_path, device="cpu"):
        self.model = YOLO(model_path)
        self.device = device

    def detect(self, frame, visualize=True):
        results = self.model.predict(
            source=frame,
            device=self.device,
            verbose=False
        )[0]

        detections = []

        vis = frame.copy()

        if results.boxes is not None:
            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])

                if hasattr(self.model, "names"):
                    cls_name = self.model.names[cls_id]
                else:
                    cls_name = str(cls_id)

                # output
                detections.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "class": cls_name,
                    "conf":   float(box.conf[0]),      
                })

                # Visulaization
                if visualize:
                    # bbox
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(vis, cls_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if visualize:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            #debug_path = f"debug/yolo_pic/yolo_debug_{timestamp}.png"

            cv2.imshow("YOLO", vis)
            cv2.waitKey(1)
            
        return detections