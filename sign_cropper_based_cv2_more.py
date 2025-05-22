# 개선된 한국 어린이보호구역 표지판 전용 크로퍼
# 다양한 형태의 표지판 탐지 (간단한 형태부터 복잡한 형태까지)

import cv2
import numpy as np
import os
from pathlib import Path
import json

class ImprovedKoreanSchoolZoneCropper:
    def __init__(self, output_dir="school_zone_signs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        print("개선된 한국 어린이보호구역 표지판 크로퍼 초기화 완료!")
        
        # 더 넓은 노란색/주황색 범위 (다양한 조명 조건 고려)
        self.yellow_ranges = [
            (np.array([10, 80, 120]), np.array([45, 255, 255])),    # 밝은 노란색~주황색
            (np.array([15, 50, 100]), np.array([35, 255, 255])),    # 표준 노란색
            (np.array([20, 40, 80]), np.array([40, 200, 255])),     # 어두운 노란색
            (np.array([8, 100, 150]), np.array([25, 255, 255]))     # 레몬 노란색
        ]
        
        # 파란색 범위 확장
        self.blue_ranges = [
            (np.array([95, 40, 40]), np.array([135, 255, 255])),    # 표준 파란색
            (np.array([100, 20, 30]), np.array([130, 255, 255])),   # 연한 파란색
            (np.array([90, 60, 60]), np.array([125, 255, 255]))     # 진한 파란색
        ]
        
        # 빨간색 범위 확장
        self.red_ranges = [
            (np.array([0, 40, 40]), np.array([15, 255, 255])),      # 빨간색 1
            (np.array([165, 40, 40]), np.array([180, 255, 255])),   # 빨간색 2
            (np.array([0, 20, 30]), np.array([20, 255, 255])),      # 연한 빨간색
            (np.array([160, 20, 30]), np.array([180, 255, 255]))    # 연한 빨간색 2
        ]
        
        # 검은색/어두운 색상 범위 (텍스트용)
        self.dark_ranges = [
            (np.array([0, 0, 0]), np.array([180, 255, 80]))         # 검은색~어두운 회색
        ]
    
    def detect_yellow_background_enhanced(self, image):
        """향상된 노란색/주황색 배경 탐지"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 원본 이미지 전처리 (조명 보정)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        hsv_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        
        yellow_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        # 원본과 향상된 이미지 모두에서 노란색 탐지
        for hsv_img in [hsv, hsv_enhanced]:
            for lower, upper in self.yellow_ranges:
                mask = cv2.inRange(hsv_img, lower, upper)
                yellow_mask = cv2.bitwise_or(yellow_mask, mask)
        
        # 가우시안 블러로 부드럽게
        yellow_mask = cv2.GaussianBlur(yellow_mask, (5, 5), 0)
        
        # 모폴로지 연산으로 정리
        kernel = np.ones((7, 7), np.uint8)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        
        # 약간 팽창 (표지판 경계까지 포함)
        yellow_mask = cv2.dilate(yellow_mask, kernel, iterations=1)
        
        return yellow_mask
    
    def detect_blue_elements(self, image, yellow_mask):
        """파란색 요소 탐지 (삼각형, 사각형 모두)"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        blue_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for lower, upper in self.blue_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            blue_mask = cv2.bitwise_or(blue_mask, mask)
        
        # 노란색 영역 내부의 파란색만
        blue_in_yellow = cv2.bitwise_and(blue_mask, yellow_mask)
        
        # 작은 노이즈 제거
        kernel = np.ones((3, 3), np.uint8)
        blue_in_yellow = cv2.morphologyEx(blue_in_yellow, cv2.MORPH_OPEN, kernel)
        blue_in_yellow = cv2.morphologyEx(blue_in_yellow, cv2.MORPH_CLOSE, kernel)
        
        return blue_in_yellow
    
    def detect_red_elements(self, image, yellow_mask):
        """빨간색 요소 탐지 (원, 링 모두)"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for lower, upper in self.red_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            red_mask = cv2.bitwise_or(red_mask, mask)
        
        # 노란색 영역 내부의 빨간색만
        red_in_yellow = cv2.bitwise_and(red_mask, yellow_mask)
        
        # 작은 노이즈 제거
        kernel = np.ones((3, 3), np.uint8)
        red_in_yellow = cv2.morphologyEx(red_in_yellow, cv2.MORPH_OPEN, kernel)
        red_in_yellow = cv2.morphologyEx(red_in_yellow, cv2.MORPH_CLOSE, kernel)
        
        return red_in_yellow
    
    def detect_text_enhanced(self, image, yellow_mask):
        """향상된 텍스트 탐지 (한글 + 영문)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 노란색 영역에서만 작업
        gray_in_yellow = cv2.bitwise_and(gray, gray, mask=yellow_mask)
        
        # 여러 방법으로 텍스트 탐지
        text_masks = []
        
        # 1. 적응적 임계값
        binary1 = cv2.adaptiveThreshold(gray_in_yellow, 255, 
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        text_masks.append(binary1)
        
        # 2. Otsu 임계값
        _, binary2 = cv2.threshold(gray_in_yellow, 0, 255, 
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_masks.append(binary2)
        
        # 3. 어두운 색상 기반 탐지
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        dark_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in self.dark_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            dark_mask = cv2.bitwise_or(dark_mask, mask)
        
        dark_in_yellow = cv2.bitwise_and(dark_mask, yellow_mask)
        text_masks.append(dark_in_yellow)
        
        # 모든 텍스트 마스크 결합
        combined_text = np.zeros_like(gray_in_yellow)
        for mask in text_masks:
            combined_text = cv2.bitwise_or(combined_text, mask)
        
        # 모폴로지로 텍스트 라인 강화
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))  # 가로
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))  # 세로
        
        h_lines = cv2.morphologyEx(combined_text, cv2.MORPH_CLOSE, kernel_h)
        v_lines = cv2.morphologyEx(combined_text, cv2.MORPH_CLOSE, kernel_v)
        
        text_final = cv2.bitwise_or(h_lines, v_lines)
        
        # 작은 노이즈 제거
        kernel_clean = np.ones((2, 2), np.uint8)
        text_final = cv2.morphologyEx(text_final, cv2.MORPH_OPEN, kernel_clean)
        
        return text_final
    
    def detect_rectangular_regions(self, image):
        """사각형 영역 탐지 (표지판 형태)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 엣지 검출
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # 직선 검출
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=30, maxLineGap=10)
        
        # 직선들로부터 사각형 추정
        line_mask = np.zeros_like(gray)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
        
        # 모폴로지로 연결
        kernel = np.ones((5, 5), np.uint8)
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel)
        
        return line_mask
    
    def combine_all_features(self, image):
        """모든 특징을 종합하여 표지판 탐지"""
        height, width = image.shape[:2]
        
        # 1. 노란색 배경 탐지 (가장 중요)
        yellow_mask = self.detect_yellow_background_enhanced(image)
        
        # 2. 파란색 요소
        blue_mask = self.detect_blue_elements(image, yellow_mask)
        
        # 3. 빨간색 요소
        red_mask = self.detect_red_elements(image, yellow_mask)
        
        # 4. 텍스트 영역
        text_mask = self.detect_text_enhanced(image, yellow_mask)
        
        # 5. 사각형 구조
        rect_mask = self.detect_rectangular_regions(image)
        rect_in_yellow = cv2.bitwise_and(rect_mask, yellow_mask)
        
        # 특징별 가중치 점수 계산
        score_map = np.zeros((height, width), dtype=np.float32)
        
        # 가중치 설정 (조건에 따라 유연하게)
        score_map += yellow_mask.astype(np.float32) * 0.4    # 노란색 배경 40%
        score_map += blue_mask.astype(np.float32) * 0.2      # 파란색 20%
        score_map += red_mask.astype(np.float32) * 0.2       # 빨간색 20%
        score_map += text_mask.astype(np.float32) * 0.15     # 텍스트 15%
        score_map += rect_in_yellow.astype(np.float32) * 0.05 # 구조 5%
        
        # 정규화
        if score_map.max() > 0:
            score_map = (score_map / score_map.max() * 255)
        
        # 다단계 임계값으로 후보 생성
        candidates_masks = []
        thresholds = [100, 120, 150]  # 여러 임계값
        
        for thresh in thresholds:
            _, candidate_mask = cv2.threshold(score_map.astype(np.uint8), 
                                            thresh, 255, cv2.THRESH_BINARY)
            
            # 정리
            kernel = np.ones((5, 5), np.uint8)
            candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel)
            candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel)
            
            candidates_masks.append(candidate_mask)
        
        return candidates_masks, score_map
    
    def extract_enhanced_candidates(self, image, masks_list, score_map):
        """향상된 후보 추출"""
        all_candidates = []
        
        for mask_idx, mask in enumerate(masks_list):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # 더 유연한 크기 조건
                if area > 500:  # 최소 크기 조건 완화
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # 더 넓은 비율 허용 (정사각형부터 세로 직사각형까지)
                    if (0.3 <= aspect_ratio <= 2.5 and 
                        w >= 30 and h >= 30 and 
                        w <= 800 and h <= 800):
                        
                        # ROI 점수 계산
                        roi_score = score_map[y:y+h, x:x+w]
                        avg_score = np.mean(roi_score) if roi_score.size > 0 else 0
                        
                        # 색상 검증
                        roi_image = image[y:y+h, x:x+w]
                        color_confidence = self.verify_school_zone_colors_enhanced(roi_image)
                        
                        # 모양 검증
                        shape_confidence = self.verify_rectangular_shape(contour)
                        
                        # 최종 신뢰도
                        final_confidence = (
                            (avg_score / 255.0) * 0.5 + 
                            color_confidence * 0.35 + 
                            shape_confidence * 0.15
                        )
                        
                        # 임계값에 따른 보너스
                        threshold_bonus = (len(masks_list) - mask_idx) * 0.05
                        final_confidence += threshold_bonus
                        
                        all_candidates.append({
                            'bbox': (x, y, x + w, y + h),
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'avg_score': avg_score,
                            'color_confidence': color_confidence,
                            'shape_confidence': shape_confidence,
                            'confidence': min(1.0, final_confidence),
                            'mask_level': mask_idx
                        })
        
        # 중복 제거 (IoU 기반)
        final_candidates = self.remove_duplicate_candidates(all_candidates)
        
        # 신뢰도 순 정렬
        final_candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        return final_candidates
    
    def remove_duplicate_candidates(self, candidates, iou_threshold=0.3):
        """IoU 기반 중복 후보 제거"""
        if not candidates:
            return []
        
        # 신뢰도순 정렬
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        
        for candidate in candidates:
            should_keep = True
            x1, y1, x2, y2 = candidate['bbox']
            
            for kept in keep:
                kx1, ky1, kx2, ky2 = kept['bbox']
                
                # IoU 계산
                intersection_x1 = max(x1, kx1)
                intersection_y1 = max(y1, ky1)
                intersection_x2 = min(x2, kx2)
                intersection_y2 = min(y2, ky2)
                
                if intersection_x1 < intersection_x2 and intersection_y1 < intersection_y2:
                    intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                    area1 = (x2 - x1) * (y2 - y1)
                    area2 = (kx2 - kx1) * (ky2 - ky1)
                    union_area = area1 + area2 - intersection_area
                    
                    iou = intersection_area / union_area if union_area > 0 else 0
                    
                    if iou > iou_threshold:
                        should_keep = False
                        break
            
            if should_keep:
                keep.append(candidate)
        
        return keep
    
    def verify_school_zone_colors_enhanced(self, roi):
        """향상된 색상 검증"""
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
        
        # 검은색/어두운 색 비율 (텍스트)
        dark_mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
        for lower, upper in self.dark_ranges:
            mask = cv2.inRange(hsv_roi, lower, upper)
            dark_mask = cv2.bitwise_or(dark_mask, mask)
        dark_ratio = cv2.countNonZero(dark_mask) / total_pixels
        
        # 점수 계산 (더 유연한 기준)
        color_score = 0.0
        
        # 노란색이 주를 이루어야 함
        if yellow_ratio > 0.2:  # 20% 이상
            color_score += 0.4
        elif yellow_ratio > 0.1:  # 10% 이상
            color_score += 0.2
        
        # 파란색 또는 빨간색이 있어야 함
        if blue_ratio > 0.02:   # 2% 이상
            color_score += 0.3
        if red_ratio > 0.01:    # 1% 이상
            color_score += 0.3
        
        # 텍스트(어두운 색)가 있으면 보너스
        if dark_ratio > 0.05:   # 5% 이상
            color_score += 0.1
        
        return min(1.0, color_score)
    
    def verify_rectangular_shape(self, contour):
        """사각형 모양 검증"""
        # 윤곽선 근사
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # 꼭짓점 개수로 모양 평가
        vertices = len(approx)
        
        if vertices == 4:  # 정사각형/직사각형
            return 1.0
        elif 3 <= vertices <= 6:  # 사각형에 가까운 모양
            return 0.7
        elif vertices > 6:  # 너무 복잡하지 않은 모양
            return 0.5
        else:
            return 0.3
    
    def crop_and_save_signs_enhanced(self, image_path, confidence_threshold=0.25):
        """향상된 표지판 탐지 및 저장"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"이미지를 읽을 수 없습니다: {image_path}")
            return 0
        
        print(f"처리 중: {os.path.basename(image_path)}")
        
        # 모든 특징 결합하여 후보 탐지
        masks_list, score_map = self.combine_all_features(image)
        candidates = self.extract_enhanced_candidates(image, masks_list, score_map)
        
        # 신뢰도 필터링
        filtered_candidates = [c for c in candidates if c['confidence'] >= confidence_threshold]
        
        if not filtered_candidates:
            print(f"  -> 어린이보호구역 표지판을 찾을 수 없습니다. (임계값: {confidence_threshold})")
            return 0
        
        # 상위 5개까지 저장
        top_candidates = filtered_candidates[:5]
        
        # 크롭 및 저장
        base_name = Path(image_path).stem
        saved_count = 0
        
        for i, candidate in enumerate(top_candidates):
            x1, y1, x2, y2 = candidate['bbox']
            
            # 패딩 추가
            padding = 15
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
                print(f"     신뢰도: {candidate['confidence']:.3f}")
                print(f"     색상 검증: {candidate['color_confidence']:.3f}")
                print(f"     모양 검증: {candidate['shape_confidence']:.3f}")
                print(f"     크기: {x2-x1}x{y2-y1}")
        
        return saved_count
    
    def process_folder_enhanced(self, input_folder, confidence_threshold=0.25):
        """폴더 내 모든 이미지 처리 (향상된 버전)"""
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
        print("향상된 탐지 알고리즘 사용")
        print()
        
        for image_file in image_files:
            try:
                crops_count = self.crop_and_save_signs_enhanced(str(image_file), confidence_threshold)
                
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
    print("=== 개선된 한국 어린이보호구역 표지판 크로퍼 ===")
    print("다양한 형태의 표지판 탐지 지원")
    print("- 표준형 (노란색 + 파란삼각형 + 빨간원)")
    print("- 간단형 (노란색 + 텍스트 + 속도제한)")
    print("- 혼합형 (다양한 조합)")
    print()
    
    # 입력 설정
    input_folder = input("이미지 폴더 경로 (Enter = ./images): ").strip()
    if not input_folder:
        input_folder = "./images"
    
    output_folder = input("출력 폴더명 (Enter = school_zone_signs): ").strip()
    if not output_folder:
        output_folder = "school_zone_signs"
    
    confidence = input("신뢰도 임계값 (0.0-1.0, Enter = 0.25): ").strip()
    try:
        confidence = float(confidence) if confidence else 0.25
    except:
        confidence = 0.25
    
    # 크로퍼 초기화 및 실행
    cropper = ImprovedKoreanSchoolZoneCropper(output_folder)
    
    if os.path.isfile(input_folder):
        print(f"단일 파일 처리: {input_folder}")
        crops_count = cropper.crop_and_save_signs_enhanced(input_folder, confidence)
        print(f"완료! {crops_count}개의 어린이보호구역 표지판을 크롭했습니다.")
    else:
        print(f"폴더 처리: {input_folder}")
        results = cropper.process_folder_enhanced(input_folder, confidence)
        
        print("\n=== 처리 결과 ===")
        print(f"처리된 이미지: {results['processed']}개")
        print(f"크롭된 표지판: {results['total_crops']}개")
        print(f"성공한 이미지: {len(results['success'])}개")
        print(f"실패한 이미지: {len(results['failed'])}개")
        
        if results['failed']:
            print("\n표지판을 찾지 못한 파일들:")
            for failed_file in results['failed'][:5]:
                print(f"  - {os.path.basename(failed_file)}")
            if len(results['failed']) > 5:
                print(f"  ... 외 {len(results['failed']) - 5}개")
    
    print(f"\n크롭된 어린이보호구역 표지판들이 '{output_folder}' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    print("필요한 패키지: pip install opencv-python numpy")
    print()
    main()