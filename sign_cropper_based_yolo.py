# 가장 확실한 표지판 자동 탐지 및 크롭 프로그램
# YOLOv8 + 다단계 필터링 방식

import cv2
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO
from typing import List, Tuple
import json

class ReliableSignCropper:
    def __init__(self, output_dir="auto_cropped_signs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 서로 다른 YOLO 모델들 (앙상블 효과)
        print("YOLO 모델들 로딩 중...")
        self.models = {
            'yolov8n': YOLO('yolov8n.pt'),    # 빠르고 가벼움
            'yolov8s': YOLO('yolov8s.pt'),    # 중간 성능
        }
        
        # COCO 데이터셋에서 표지판 관련 클래스들
        self.target_classes = {
            11: 'stop_sign',           # 정지 표지판 (가장 확실)
            # 9: 'traffic_light',      # 신호등 (너무 클 수 있어서 제외)
            # 추가로 사용할 수 있는 클래스들
        }
        
        # 어린이보호구역 표지판 특징 (한국 기준)
        self.school_zone_features = {
            'colors': {
                'yellow': [(20, 100, 100), (30, 255, 255)],  # 노란색 배경
                'green': [(35, 50, 50), (85, 255, 255)],     # 초록색 테두리
            },
            'text_areas': True,  # 텍스트 영역 포함
            'size_range': (30, 300),  # 최소-최대 크기
        }
    
    def detect_with_yolo_ensemble(self, image_path: str) -> List[Tuple]:
        """여러 YOLO 모델을 사용한 앙상블 탐지"""
        image = cv2.imread(image_path)
        if image is None:
            return []
        
        all_detections = []
        
        for model_name, model in self.models.items():
            try:
                results = model(image, verbose=False, conf=0.3)  # 낮은 신뢰도로 시작
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # 표지판 관련 클래스 또는 특정 조건
                            if (class_id in self.target_classes or 
                                self.is_potential_sign(image, x1, y1, x2, y2, confidence)):
                                
                                all_detections.append({
                                    'bbox': (x1, y1, x2, y2),
                                    'confidence': confidence,
                                    'class_id': class_id,
                                    'model': model_name,
                                    'class_name': self.target_classes.get(class_id, f'unknown_{class_id}')
                                })
            except Exception as e:
                print(f"모델 {model_name} 오류: {e}")
                continue
        
        return all_detections
    
    def is_potential_sign(self, image, x1, y1, x2, y2, confidence) -> bool:
        """표지판일 가능성이 있는지 추가 검증"""
        # 1. 크기 검증
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        if not (30 <= width <= 400 and 30 <= height <= 400):
            return False
        
        if area < 900:  # 너무 작으면 제외
            return False
        
        # 2. 종횡비 검증 (표지판은 보통 정사각형이나 원형)
        aspect_ratio = width / height
        if not (0.5 <= aspect_ratio <= 2.0):
            return False
        
        # 3. 신뢰도가 낮더라도 표지판 특징이 있으면 포함
        if confidence > 0.4:
            return True
        
        # 4. 색상 기반 추가 검증
        roi = image[y1:y2, x1:x2]
        if self.has_sign_colors(roi):
            return True
        
        # 5. 에지 밀도 검증 (표지판은 에지가 많음)
        if self.has_dense_edges(roi):
            return True
        
        return False
    
    def has_sign_colors(self, roi) -> bool:
        """표지판 특유의 색상이 있는지 검사"""
        if roi.size == 0:
            return False
        
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 한국 표지판 주요 색상들
        color_ranges = {
            'red': [(0, 50, 50), (10, 255, 255), (170, 50, 50), (180, 255, 255)],
            'blue': [(100, 50, 50), (130, 255, 255)],
            'yellow': [(15, 100, 100), (35, 255, 255)],  # 어린이보호구역
            'green': [(35, 50, 50), (85, 255, 255)],
            'white': [(0, 0, 200), (180, 30, 255)],
        }
        
        total_pixels = roi.shape[0] * roi.shape[1]
        
        for color_name, ranges in color_ranges.items():
            color_pixels = 0
            
            if color_name == 'red':  # 빨간색은 두 범위
                mask1 = cv2.inRange(hsv_roi, ranges[0], ranges[1])
                mask2 = cv2.inRange(hsv_roi, ranges[2], ranges[3])
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv_roi, ranges[0], ranges[1])
            
            color_pixels = cv2.countNonZero(mask)
            color_ratio = color_pixels / total_pixels
            
            # 특정 색상이 10% 이상이면 표지판 가능성 높음
            if color_ratio > 0.1:
                return True
        
        return False
    
    def has_dense_edges(self, roi) -> bool:
        """에지 밀도가 높은지 검사 (텍스트나 심볼 존재)"""
        if roi.size == 0:
            return False
        
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_roi, 50, 150)
        
        edge_pixels = cv2.countNonZero(edges)
        total_pixels = roi.shape[0] * roi.shape[1]
        edge_density = edge_pixels / total_pixels
        
        # 에지 밀도가 5% 이상이면 텍스트/심볼 있을 가능성
        return edge_density > 0.05
    
    def filter_and_merge_detections(self, detections: List[dict]) -> List[dict]:
        """중복 탐지 제거 및 최적 탐지 선택"""
        if not detections:
            return []
        
        # NMS (Non-Maximum Suppression) 적용
        boxes = []
        scores = []
        indices = []
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            boxes.append([x1, y1, x2 - x1, y2 - y1])  # [x, y, w, h] 형식
            scores.append(det['confidence'])
            indices.append(i)
        
        if not boxes:
            return []
        
        # OpenCV NMS 적용
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        
        nms_indices = cv2.dnn.NMSBoxes(boxes, scores, 0.3, 0.4)
        
        filtered_detections = []
        if len(nms_indices) > 0:
            for i in nms_indices.flatten():
                filtered_detections.append(detections[i])
        
        return filtered_detections
    
    def crop_and_save_signs(self, image_path: str, save_metadata=True) -> int:
        """표지판을 탐지하고 크롭하여 저장"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"이미지를 읽을 수 없습니다: {image_path}")
            return 0
        
        # 1. 탐지 실행
        detections = self.detect_with_yolo_ensemble(image_path)
        
        # 2. 필터링 및 중복 제거
        filtered_detections = self.filter_and_merge_detections(detections)
        
        if not filtered_detections:
            print(f"표지판을 찾을 수 없습니다: {os.path.basename(image_path)}")
            return 0
        
        # 3. 크롭 및 저장
        base_name = Path(image_path).stem
        saved_count = 0
        metadata = []
        
        for i, detection in enumerate(filtered_detections):
            x1, y1, x2, y2 = detection['bbox']
            
            # 크롭 영역을 약간 확장 (패딩 추가)
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)
            
            # 크롭
            cropped = image[y1:y2, x1:x2]
            
            if cropped.size > 0:
                # 저장 파일명
                output_filename = f"{base_name}_sign_{i+1:02d}.jpg"
                output_path = self.output_dir / output_filename
                
                # 저장
                cv2.imwrite(str(output_path), cropped)
                saved_count += 1
                
                # 메타데이터 수집
                metadata.append({
                    'filename': output_filename,
                    'original_bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'class_name': detection['class_name'],
                    'model': detection['model'],
                    'size': f"{x2-x1}x{y2-y1}"
                })
                
                print(f"저장: {output_filename} (신뢰도: {detection['confidence']:.2f})")
        
        # 메타데이터 저장
        if save_metadata and metadata:
            metadata_file = self.output_dir / f"{base_name}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return saved_count
    
    def process_folder(self, input_folder: str) -> dict:
        """폴더 내 모든 이미지 처리"""
        input_path = Path(input_folder)
        if not input_path.exists():
            print(f"폴더가 존재하지 않습니다: {input_folder}")
            return {}
        
        # 이미지 파일 확장자
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        results = {
            'processed': 0,
            'total_crops': 0,
            'failed': [],
            'success': []
        }
        
        # 모든 이미지 파일 처리
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        print(f"총 {len(image_files)}개의 이미지를 처리합니다...")
        
        for image_file in image_files:
            try:
                crops_count = self.crop_and_save_signs(str(image_file))
                
                results['processed'] += 1
                results['total_crops'] += crops_count
                
                if crops_count > 0:
                    results['success'].append(str(image_file))
                else:
                    results['failed'].append(str(image_file))
                    
            except Exception as e:
                print(f"처리 실패 {image_file}: {e}")
                results['failed'].append(str(image_file))
        
        return results

def main():
    print("=== 가장 확실한 표지판 자동 크롭 프로그램 ===")
    print("YOLOv8 앙상블 + 다단계 필터링 방식")
    print()
    
    # 입력 폴더 설정
    input_folder = input("이미지 폴더 경로를 입력하세요 (Enter = ./images): ").strip()
    if not input_folder:
        input_folder = "./images"
    
    output_folder = input("출력 폴더명을 입력하세요 (Enter = auto_cropped_signs): ").strip()
    if not output_folder:
        output_folder = "auto_cropped_signs"
    
    # 크로퍼 초기화
    cropper = ReliableSignCropper(output_folder)
    
    # 단일 파일 vs 폴더 처리 선택
    if os.path.isfile(input_folder):
        print(f"단일 파일 처리: {input_folder}")
        crops_count = cropper.crop_and_save_signs(input_folder)
        print(f"완료! {crops_count}개의 표지판을 크롭했습니다.")
    else:
        print(f"폴더 처리: {input_folder}")
        results = cropper.process_folder(input_folder)
        
        print("\n=== 처리 결과 ===")
        print(f"처리된 이미지: {results['processed']}개")
        print(f"크롭된 표지판: {results['total_crops']}개")
        print(f"성공한 이미지: {len(results['success'])}개")
        print(f"실패한 이미지: {len(results['failed'])}개")
        
        if results['failed']:
            print("\n실패한 파일들:")
            for failed_file in results['failed'][:5]:  # 최대 5개만 표시
                print(f"  - {os.path.basename(failed_file)}")
    
    print(f"\n크롭된 이미지들이 '{output_folder}' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    print("필요한 패키지들:")
    print("pip install ultralytics opencv-python")
    print()
    
    main()