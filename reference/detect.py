"""
YOLO를 사용하여 비디오에서 사람만 감지하는 스크립트
"""

from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO("./models/yolov26n.pt")
