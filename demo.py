# 어린이보보구역 표지판 탐지 - YOLOv5 + Luxonis OAK
# 라벨이 이미지인 경우 (템플릿 매칭 방식)

import os
import cv2
import numpy as np
import torch
import yaml
from pathlib import Path
import depthai as dai
from ultralytics import YOLO
import json

# ===== 1. 템플릿 매칭을 이용한 자동 라벨링 =====

class SchoolZoneAutoLabeler:
    def __init__(self, dataset_path="school_zone_dataset"):
        self.dataset_path = Path(dataset_path)
        self.template_images = []
        self.setup_directories()
    
    def setup_directories(self):
        """디렉토리 구조 생성"""
        dirs = [
            self.dataset_path / "images" / "train",
            self.dataset_path / "labels" / "train",
            self.dataset_path / "templates"  # 템플릿 이미지들
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_template_images(self, template_folder):
        """어린이보호구역 표지판 템플릿 이미지들 로드"""
        template_path = Path(template_folder)
        self.template_images = []
        
        for img_file in template_path.glob("*.jpg") or template_path.glob("*.png"):
            template = cv2.imread(str(img_file))
            if template is not None:
                # 다양한 크기로 템플릿 준비
                for scale in [0.5, 0.75, 1.0, 1.25, 1.5]:
                    h, w = template.shape[:2]
                    new_h, new_w = int(h * scale), int(w * scale)
                    resized_template = cv2.resize(template, (new_w, new_h))
                    self.template_images.append({
                        'image': resized_template,
                        'scale': scale,
                        'name': img_file.stem
                    })
        
        print(f"총 {len(self.template_images)}개의 템플릿을 로드했습니다.")
    
    def find_sign_in_image(self, image, threshold=0.7):
        """이미지에서 어린이보호구역 표지판 찾기"""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = []
        
        for template_info in self.template_images:
            template = template_info['image']
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            
            # 템플릿 매칭
            result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= threshold)
            
            for pt in zip(*locations[::-1]):
                h, w = template.shape[:2]
                x1, y1 = pt
                x2, y2 = x1 + w, y1 + h
                confidence = result[y1, x1]
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'template': template_info['name']
                })
        
        # NMS (Non-Maximum Suppression) 적용
        detections = self.apply_nms(detections)
        return detections
    
    def apply_nms(self, detections, iou_threshold=0.3):
        """겹치는 탐지 결과 제거"""
        if not detections:
            return []
        
        # 신뢰도 순으로 정렬
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        filtered_detections = []
        for detection in detections:
            is_duplicate = False
            for filtered in filtered_detections:
                if self.calculate_iou(detection['bbox'], filtered['bbox']) > iou_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_detections.append(detection)
        
        return filtered_detections
    
    def calculate_iou(self, bbox1, bbox2):
        """IoU (Intersection over Union) 계산"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 교집합 영역
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def convert_to_yolo_format(self, image_shape, bbox):
        """바운딩박스를 YOLO 형식으로 변환"""
        h, w = image_shape[:2]
        x1, y1, x2, y2 = bbox
        
        x_center = (x1 + x2) / 2.0 / w
        y_center = (y1 + y2) / 2.0 / h
        width = (x2 - x1) / w
        height = (y2 - y1) / h
        
        return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    
    def auto_label_dataset(self, image_folder, template_folder, confidence_threshold=0.7):
        """자동으로 데이터셋 라벨링"""
        # 템플릿 이미지 로드
        self.load_template_images(template_folder)
        
        image_path = Path(image_folder)
        labeled_count = 0
        
        for img_file in image_path.glob("*.jpg") or image_path.glob("*.png"):
            print(f"Processing: {img_file.name}")
            
            # 이미지 로드
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            
            # 표지판 탐지
            detections = self.find_sign_in_image(image, confidence_threshold)
            
            if detections:
                # train 폴더에 이미지 복사
                train_img_path = self.dataset_path / "images" / "train" / img_file.name
                cv2.imwrite(str(train_img_path), image)
                
                # YOLO 형식으로 라벨 생성
                label_file = self.dataset_path / "labels" / "train" / f"{img_file.stem}.txt"
                with open(label_file, 'w') as f:
                    for detection in detections:
                        yolo_label = self.convert_to_yolo_format(image.shape, detection['bbox'])
                        f.write(yolo_label + '\n')
                
                labeled_count += 1
                print(f"  -> {len(detections)}개의 표지판 발견")
            else:
                print("  -> 표지판 없음")
        
        print(f"\n총 {labeled_count}개의 이미지가 라벨링되었습니다.")
        return labeled_count

# ===== 2. Train/Val 분할 없이 학습 =====

class SchoolZoneTrainerSimple:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.model = None
        
    def create_dataset_yaml(self):
        """Train만 있는 dataset.yaml 생성"""
        # train 데이터의 일부를 val로 자동 분할
        dataset_config = {
            'path': str(self.dataset_path.absolute()),
            'train': 'images/train',
            'val': 'images/train',  # train과 동일하게 설정 (YOLO가 자동으로 분할)
            'nc': 1,
            'names': ['school_zone_sign']
        }
        
        yaml_path = self.dataset_path / "dataset.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        return yaml_path
    
    def train_model(self, epochs=100, img_size=640, batch_size=8):
        """YOLOv5 모델 학습"""
        yaml_path = self.create_dataset_yaml()
        
        print("YOLOv5 모델 학습 시작...")
        
        # YOLOv5n (nano) 모델 사용 (경량화)
        self.model = YOLO('yolov5n.pt')
        
        # 학습 실행
        results = self.model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            name='school_zone_detection',
            project='runs/train',
            val=0.2,  # 20%를 validation으로 자동 분할
            patience=10,  # 조기 종료
            save_period=10  # 10 에포크마다 저장
        )
        
        print("학습 완료!")
        return results

# ===== 3. 실시간 탐지 (이전과 동일) =====

class SchoolZoneDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            # 학습된 모델이 없으면 기본 YOLO 모델 사용
            self.model = YOLO('yolov5n.pt')
            self.use_pretrained = True
        else:
            self.model_path = model_path
            self.use_pretrained = False
        
        self.pipeline = None
        self.device = None
        
    def create_pipeline_cpu(self):
        """CPU에서 실행하는 파이프라인 (OAK 없이 테스트용)"""
        cap = cv2.VideoCapture(0)  # 웹캠 사용
        
        print("어린이보호구역 표지판 탐지 시작 (q를 눌러 종료)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # YOLO 추론
            if self.use_pretrained:
                # 사전 훈련된 모델로 일반 객체 탐지
                results = self.model(frame, verbose=False)
                self.draw_pretrained_detections(frame, results)
            else:
                # 커스텀 학습된 모델 사용
                results = self.model(frame, verbose=False)
                self.draw_custom_detections(frame, results)
            
            cv2.imshow("어린이보호구역 표지판 탐지", frame)
            
            if cv2.waitKey(1) == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def draw_pretrained_detections(self, frame, results):
        """사전 훈련된 모델 결과 표시 (stop sign 등)"""
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 클래스 11은 stop sign (유사한 표지판으로 대체)
                    if int(box.cls[0]) == 11:  # stop sign
                        x1, y1, x2, y2 = box.xyxy[0]
                        confidence = box.conf[0]
                        
                        # 바운딩박스 그리기
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        
                        label = f"Sign Detected: {confidence:.2f}"
                        cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def draw_custom_detections(self, frame, results):
        """커스텀 모델 결과 표시"""
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    confidence = box.conf[0]
                    
                    # 바운딩박스 그리기
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    label = f"School Zone: {confidence:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # 경고 메시지
                    if confidence > 0.7:
                        cv2.putText(frame, "어린이보호구역 감지!", (50, 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

# ===== 메인 실행 코드 =====

def main():
    print("=== 어린이보호구역 표지판 탐지 시스템 ===")
    print("라벨 이미지를 사용한 자동 라벨링 + 학습 + 탐지")
    
    # 폴더 경로 설정
    image_folder = input("학습용 이미지 폴더 경로를 입력하세요: ").strip()
    template_folder = input("표지판 템플릿 이미지 폴더 경로를 입력하세요: ").strip()
    
    if not image_folder or not template_folder:
        print("기본 경로 사용: ./images, ./templates")
        image_folder = "./images"
        template_folder = "./targets"
    
    # 1. 자동 라벨링
    print("\n1. 자동 라벨링 중...")
    labeler = SchoolZoneAutoLabeler()
    
    if os.path.exists(template_folder):
        labeled_count = labeler.auto_label_dataset(image_folder, template_folder)
        
        if labeled_count == 0:
            print("라벨링된 이미지가 없습니다. 템플릿 매칭 임계값을 낮춰보세요.")
            return
    else:
        print(f"템플릿 폴더 '{template_folder}'가 없습니다.")
        print("수동으로 라벨링하거나 템플릿 이미지를 준비해주세요.")
        return
    
    # 2. 모델 학습
    user_input = input(f"\n{labeled_count}개 이미지로 학습을 진행하시겠습니까? (y/n): ")
    if user_input.lower() == 'y':
        print("\n2. 모델 학습 중...")
        trainer = SchoolZoneTrainerSimple("school_zone_dataset")
        trainer.train_model(epochs=50, batch_size=4)  # 작은 데이터셋용 설정
        model_path = "runs/train/school_zone_detection/weights/best.pt"
    else:
        model_path = None
    
    # 3. 실시간 탐지 (CPU에서)
    user_input = input("\n실시간 탐지를 시작하시겠습니까? (y/n): ")
    if user_input.lower() == 'y':
        print("\n3. 실시간 탐지 시작...")
        detector = SchoolZoneDetector(model_path)
        detector.create_pipeline_cpu()

if __name__ == "__main__":
    print("필요한 패키지들:")
    print("pip install ultralytics opencv-python torch")
    print()
    
    print("폴더 구조 예시:")
    print("./images/          # 어린이보호구역이 포함된 전체 장면 이미지들")
    print("./templates/       # 어린이보호구역 표지판만 크롭된 이미지들")
    print()
    
    main()