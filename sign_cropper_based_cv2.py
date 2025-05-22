# 한국 어린이보호구역 표지판 전용 크로퍼
# 노란색 배경 + 파란 삼각형 + "30" 속도제한 특화

import cv2
import numpy as np
import os
from pathlib import Path
import json

class KoreanSchoolZoneCropper:
    def __init__(self, output_dir="school_zone_signs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        print("한국 어린이보호구역 표지판 전용 크로퍼 초기화 완료!")
        
        # 어린이보호구역 특화 설정
        self.yellow_ranges = [
            (np.array([15, 100, 100]), np.array([35, 255, 255])),  # 밝은 노란색
            (np.array([10, 80, 150]), np.array([40, 255, 255]))    # 더 넓은 노란색 범위
        ]
        
        self.blue_ranges = [
            (np.array([100, 50, 50]), np.array([130, 255, 255]))   # 파란색 삼각형
        ]
        
        self.red_ranges = [
            (np.array([0, 50, 50]), np.array([10, 255, 255])),     # 빨간색 원
            (np.array([170, 50, 50]), np.array([180, 255, 255]))
        ]
    
    def detect_yellow_background(self, image):
        """노란색 배경 탐지 (어린이보호구역의 핵심 특징)"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        yellow_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for lower, upper in self.yellow_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            yellow_mask = cv2.bitwise_or(yellow_mask, mask)
        
        # 노이즈 제거 및 영역 확장
        kernel = np.ones((5, 5), np.uint8)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        yellow_mask = cv2.dilate(yellow_mask, kernel, iterations=2)
        
        return yellow_mask
    
    def detect_blue_triangle(self, image, yellow_mask):
        """파란색 삼각형 탐지 (어린이 실루엣 영역)"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        blue_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for lower, upper in self.blue_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            blue_mask = cv2.bitwise_or(blue_mask, mask)
        
        # 노란색 영역 내부의 파란색만 추출
        blue_in_yellow = cv2.bitwise_and(blue_mask, yellow_mask)
        
        # 삼각형 모양 필터링
        contours, _ = cv2.findContours(blue_in_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        triangle_mask = np.zeros(blue_mask.shape, dtype=np.uint8)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200:  # 최소 크기
                # 윤곽선 근사하여 삼각형인지 확인
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                # 삼각형이거나 삼각형에 가까운 모양
                if len(approx) >= 3 and len(approx) <= 6:
                    cv2.drawContours(triangle_mask, [contour], -1, 255, -1)
        
        return triangle_mask
    
    def detect_red_circle_with_30(self, image, yellow_mask):
        """빨간 원 안의 "30" 숫자 탐지"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for lower, upper in self.red_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            red_mask = cv2.bitwise_or(red_mask, mask)
        
        # 노란색 영역 내부의 빨간색만 추출
        red_in_yellow = cv2.bitwise_and(red_mask, yellow_mask)
        
        # 원형 모양 필터링
        contours, _ = cv2.findContours(red_in_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circle_mask = np.zeros(red_mask.shape, dtype=np.uint8)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # 최소 크기
                # 원형도 검사
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.5:  # 어느 정도 원형
                        cv2.drawContours(circle_mask, [contour], -1, 255, -1)
        
        return circle_mask
    
    def detect_school_zone_text(self, image, yellow_mask):
        """어린이보호구역/SCHOOL ZONE 텍스트 탐지"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 노란색 영역에서만 텍스트 찾기
        gray_in_yellow = cv2.bitwise_and(gray, gray, mask=yellow_mask)
        
        # 적응적 임계값으로 텍스트 강조
        binary = cv2.adaptiveThreshold(gray_in_yellow, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # 세로 방향 모폴로지 (한글 특성)
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_vertical)
        
        # 가로 방향 모폴로지 (영문 특성)
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_horizontal)
        
        # 텍스트 영역 합치기
        text_mask = cv2.bitwise_or(vertical_lines, horizontal_lines)
        
        # 작은 노이즈 제거
        kernel_clean = np.ones((2, 2), np.uint8)
        text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN, kernel_clean)
        
        return text_mask
    
    def combine_school_zone_features(self, image):
        """어린이보호구역 표지판의 모든 특징 결합"""
        # 1. 노란색 배경 탐지 (가장 중요)
        yellow_mask = self.detect_yellow_background(image)
        
        # 2. 파란색 삼각형 탐지
        blue_triangle = self.detect_blue_triangle(image, yellow_mask)
        
        # 3. 빨간색 원 (30 속도제한) 탐지
        red_circle = self.detect_red_circle_with_30(image, yellow_mask)
        
        # 4. 텍스트 영역 탐지
        text_mask = self.detect_school_zone_text(image, yellow_mask)
        
        # 특징들을 가중치로 결합
        combined_score = np.zeros_like(yellow_mask, dtype=np.float32)
        
        # 가중치 설정
        combined_score += yellow_mask.astype(np.float32) * 0.4      # 노란색 배경 40%
        combined_score += blue_triangle.astype(np.float32) * 0.25   # 파란 삼각형 25%
        combined_score += red_circle.astype(np.float32) * 0.25      # 빨간 원 25%
        combined_score += text_mask.astype(np.float32) * 0.1        # 텍스트 10%
        
        # 점수를 0-255로 정규화
        if combined_score.max() > 0:
            combined_score = (combined_score / combined_score.max() * 255).astype(np.uint8)
        else:
            combined_score = combined_score.astype(np.uint8)
        
        # 임계값 적용 (50% 이상 점수)
        _, final_mask = cv2.threshold(combined_score, 127, 255, cv2.THRESH_BINARY)
        
        # 최종 정리
        kernel = np.ones((7, 7), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        final_mask = cv2.dilate(final_mask, kernel, iterations=1)
        
        return final_mask, combined_score
    
    def extract_school_zone_candidates(self, image, mask, score_map):
        """어린이보호구역 표지판 후보 추출"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # 최소 크기 (표지판은 어느 정도 커야 함)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # 어린이보호구역 표지판 비율 (보통 세로가 더 긴 직사각형)
                if (0.5 <= aspect_ratio <= 1.8 and 
                    w >= 50 and h >= 60 and 
                    w <= 500 and h <= 600):
                    
                    # ROI에서 평균 점수 계산
                    roi_score = score_map[y:y+h, x:x+w]
                    avg_score = np.mean(roi_score) if roi_score.size > 0 else 0
                    
                    # 색상 특징 검증
                    roi_image = image[y:y+h, x:x+w]
                    color_confidence = self.verify_school_zone_colors(roi_image)
                    
                    # 최종 신뢰도 계산
                    final_confidence = (avg_score / 255.0) * 0.7 + color_confidence * 0.3
                    
                    candidates.append({
                        'bbox': (x, y, x + w, y + h),
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'avg_score': avg_score,
                        'color_confidence': color_confidence,
                        'confidence': final_confidence
                    })
        
        # 신뢰도 순으로 정렬
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        return candidates
    
    def verify_school_zone_colors(self, roi):
        """ROI가 어린이보호구역 색상 조합을 가지고 있는지 검증"""
        if roi.size == 0:
            return 0.0
        
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        total_pixels = roi.shape[0] * roi.shape[1]
        
        # 노란색 비율
        yellow_mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
        for lower, upper in self.yellow_ranges:
            mask = cv2.inRange(hsv_roi, lower, upper)
            yellow_mask = cv2.bitwise_or(yellow_mask, mask)
        yellow_ratio = cv2.countNonZero(yellow_mask) / total_pixels
        
        # 파란색 비율
        blue_mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
        for lower, upper in self.blue_ranges:
            mask = cv2.inRange(hsv_roi, lower, upper)
            blue_mask = cv2.bitwise_or(blue_mask, mask)
        blue_ratio = cv2.countNonZero(blue_mask) / total_pixels
        
        # 빨간색 비율
        red_mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
        for lower, upper in self.red_ranges:
            mask = cv2.inRange(hsv_roi, lower, upper)
            red_mask = cv2.bitwise_or(red_mask, mask)
        red_ratio = cv2.countNonZero(red_mask) / total_pixels
        
        # 어린이보호구역 색상 조합 점수
        # 노란색이 주를 이루고, 파란색과 빨간색이 적절히 있어야 함
        color_score = 0.0
        
        if yellow_ratio > 0.3:  # 노란색이 30% 이상
            color_score += 0.5
        
        if blue_ratio > 0.05:   # 파란색이 5% 이상
            color_score += 0.25
        
        if red_ratio > 0.02:    # 빨간색이 2% 이상
            color_score += 0.25
        
        return min(1.0, color_score)
    
    def crop_and_save_signs(self, image_path, confidence_threshold=0.3):
        """이미지에서 어린이보호구역 표지판을 탐지하고 크롭하여 저장"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"이미지를 읽을 수 없습니다: {image_path}")
            return 0
        
        print(f"처리 중: {os.path.basename(image_path)}")
        
        # 어린이보호구역 특징 탐지
        mask, score_map = self.combine_school_zone_features(image)
        candidates = self.extract_school_zone_candidates(image, mask, score_map)
        
        # 신뢰도 필터링
        filtered_candidates = [c for c in candidates if c['confidence'] >= confidence_threshold]
        
        if not filtered_candidates:
            print(f"  -> 어린이보호구역 표지판을 찾을 수 없습니다. (임계값: {confidence_threshold})")
            return 0
        
        # 상위 3개만 저장 (중복 방지)
        top_candidates = filtered_candidates[:3]
        
        # 크롭 및 저장
        base_name = Path(image_path).stem
        saved_count = 0
        
        for i, candidate in enumerate(top_candidates):
            x1, y1, x2, y2 = candidate['bbox']
            
            # 패딩 추가 (표지판 전체가 보이도록)
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)
            
            # 크롭
            cropped = image[y1:y2, x1:x2]
            
            if cropped.size > 0:
                # 저장
                output_filename = f"{base_name}_school_zone_{i+1:02d}.jpg"
                output_path = self.output_dir / output_filename
                cv2.imwrite(str(output_path), cropped)
                saved_count += 1
                
                print(f"  -> 저장: {output_filename}")
                print(f"     신뢰도: {candidate['confidence']:.2f}")
                print(f"     색상 검증: {candidate['color_confidence']:.2f}")
                print(f"     크기: {x2-x1}x{y2-y1}")
        
        return saved_count
    
    def process_folder(self, input_folder, confidence_threshold=0.3):
        """폴더 내 모든 이미지 처리"""
        input_path = Path(input_folder)
        if not input_path.exists():
            print(f"폴더가 존재하지 않습니다: {input_folder}")
            return {}
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        results = {
            'processed': 0,
            'total_crops': 0,
            'failed': [],
            'success': []
        }
        
        print(f"총 {len(image_files)}개의 이미지를 처리합니다...")
        print(f"신뢰도 임계값: {confidence_threshold}")
        print()
        
        for image_file in image_files:
            try:
                crops_count = self.crop_and_save_signs(str(image_file), confidence_threshold)
                
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
    print("=== 한국 어린이보호구역 표지판 전용 크로퍼 ===")
    print("노란색 배경 + 파란 삼각형 + 빨간 원(30) 특화 탐지")
    print()
    
    # 입력 설정
    input_folder = input("이미지 폴더 경로 (Enter = ./images): ").strip()
    if not input_folder:
        input_folder = "./images"
    
    output_folder = input("출력 폴더명 (Enter = school_zone_signs): ").strip()
    if not output_folder:
        output_folder = "school_zone_signs"
    
    confidence = input("신뢰도 임계값 (0.0-1.0, Enter = 0.3): ").strip()
    try:
        confidence = float(confidence) if confidence else 0.3
    except:
        confidence = 0.3
    
    # 크로퍼 초기화 및 실행
    cropper = KoreanSchoolZoneCropper(output_folder)
    
    if os.path.isfile(input_folder):
        print(f"단일 파일 처리: {input_folder}")
        crops_count = cropper.crop_and_save_signs(input_folder, confidence)
        print(f"완료! {crops_count}개의 어린이보호구역 표지판을 크롭했습니다.")
    else:
        print(f"폴더 처리: {input_folder}")
        results = cropper.process_folder(input_folder, confidence)
        
        print("\n=== 처리 결과 ===")
        print(f"처리된 이미지: {results['processed']}개")
        print(f"크롭된 표지판: {results['total_crops']}개")
        print(f"성공한 이미지: {len(results['success'])}개")
        print(f"실패한 이미지: {len(results['failed'])}개")
        
        if results['failed']:
            print("\n표지판을 찾지 못한 파일들:")
            for failed_file in results['failed'][:5]:
                print(f"  - {os.path.basename(failed_file)}")
    
    print(f"\n크롭된 어린이보호구역 표지판들이 '{output_folder}' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    print("필요한 패키지: pip install opencv-python numpy")
    print()
    main()