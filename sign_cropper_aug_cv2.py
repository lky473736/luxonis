# 한국 어린이보호구역 표지판 크로핑 + 데이터 증강 통합 시스템 (통합 폴더 구조)
import cv2
import numpy as np
import os
from pathlib import Path
import json
import shutil

class UnifiedSchoolZoneProcessor:
    def __init__(self, data_dir="data", target_dir="target"):
        self.data_dir = Path(data_dir)  # 원본 이미지 + 증강
        self.target_dir = Path(target_dir)  # 크롭된 표지판 + 증강
        
        # 폴더 생성
        self.data_dir.mkdir(exist_ok=True)
        self.target_dir.mkdir(exist_ok=True)
        
        print("통합 폴더 구조 어린이보호구역 표지판 처리 시스템 초기화 완료!")
        print(f"원본 데이터 저장: {self.data_dir}/")
        print(f"크롭 데이터 저장: {self.target_dir}/")
        
        # 어린이보호구역 특화 설정
        self.yellow_ranges = [
            (np.array([10, 80, 120]), np.array([45, 255, 255])),
            (np.array([15, 50, 100]), np.array([35, 255, 255])),
            (np.array([20, 40, 80]), np.array([40, 200, 255])),
            (np.array([8, 100, 150]), np.array([25, 255, 255]))
        ]
        
        self.blue_ranges = [
            (np.array([95, 40, 40]), np.array([135, 255, 255])),
            (np.array([100, 20, 30]), np.array([130, 255, 255])),
            (np.array([90, 60, 60]), np.array([125, 255, 255]))
        ]
        
        self.red_ranges = [
            (np.array([0, 40, 40]), np.array([15, 255, 255])),
            (np.array([165, 40, 40]), np.array([180, 255, 255])),
            (np.array([0, 20, 30]), np.array([20, 255, 255])),
            (np.array([160, 20, 30]), np.array([180, 255, 255]))
        ]
        
        self.dark_ranges = [
            (np.array([0, 0, 0]), np.array([180, 255, 80]))
        ]
    
    # ========== 크로핑 관련 메서드들 ==========
    def detect_yellow_background_enhanced(self, image):
        """향상된 노란색/주황색 배경 탐지"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 조명 보정
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        hsv_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        
        yellow_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for hsv_img in [hsv, hsv_enhanced]:
            for lower, upper in self.yellow_ranges:
                mask = cv2.inRange(hsv_img, lower, upper)
                yellow_mask = cv2.bitwise_or(yellow_mask, mask)
        
        yellow_mask = cv2.GaussianBlur(yellow_mask, (5, 5), 0)
        kernel = np.ones((7, 7), np.uint8)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        yellow_mask = cv2.dilate(yellow_mask, kernel, iterations=1)
        
        return yellow_mask
    
    def detect_blue_elements(self, image, yellow_mask):
        """파란색 요소 탐지"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blue_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for lower, upper in self.blue_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            blue_mask = cv2.bitwise_or(blue_mask, mask)
        
        blue_in_yellow = cv2.bitwise_and(blue_mask, yellow_mask)
        kernel = np.ones((3, 3), np.uint8)
        blue_in_yellow = cv2.morphologyEx(blue_in_yellow, cv2.MORPH_OPEN, kernel)
        blue_in_yellow = cv2.morphologyEx(blue_in_yellow, cv2.MORPH_CLOSE, kernel)
        
        return blue_in_yellow
    
    def detect_red_elements(self, image, yellow_mask):
        """빨간색 요소 탐지"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for lower, upper in self.red_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            red_mask = cv2.bitwise_or(red_mask, mask)
        
        red_in_yellow = cv2.bitwise_and(red_mask, yellow_mask)
        kernel = np.ones((3, 3), np.uint8)
        red_in_yellow = cv2.morphologyEx(red_in_yellow, cv2.MORPH_OPEN, kernel)
        red_in_yellow = cv2.morphologyEx(red_in_yellow, cv2.MORPH_CLOSE, kernel)
        
        return red_in_yellow
    
    def detect_text_enhanced(self, image, yellow_mask):
        """향상된 텍스트 탐지"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_in_yellow = cv2.bitwise_and(gray, gray, mask=yellow_mask)
        
        text_masks = []
        
        # 적응적 임계값
        binary1 = cv2.adaptiveThreshold(gray_in_yellow, 255, 
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        text_masks.append(binary1)
        
        # Otsu 임계값
        _, binary2 = cv2.threshold(gray_in_yellow, 0, 255, 
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_masks.append(binary2)
        
        # 어두운 색상 기반
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        dark_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in self.dark_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            dark_mask = cv2.bitwise_or(dark_mask, mask)
        
        dark_in_yellow = cv2.bitwise_and(dark_mask, yellow_mask)
        text_masks.append(dark_in_yellow)
        
        combined_text = np.zeros_like(gray_in_yellow)
        for mask in text_masks:
            combined_text = cv2.bitwise_or(combined_text, mask)
        
        # 모폴로지
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        
        h_lines = cv2.morphologyEx(combined_text, cv2.MORPH_CLOSE, kernel_h)
        v_lines = cv2.morphologyEx(combined_text, cv2.MORPH_CLOSE, kernel_v)
        
        text_final = cv2.bitwise_or(h_lines, v_lines)
        kernel_clean = np.ones((2, 2), np.uint8)
        text_final = cv2.morphologyEx(text_final, cv2.MORPH_OPEN, kernel_clean)
        
        return text_final
    
    def combine_all_features(self, image):
        """모든 특징 결합"""
        height, width = image.shape[:2]
        
        yellow_mask = self.detect_yellow_background_enhanced(image)
        blue_mask = self.detect_blue_elements(image, yellow_mask)
        red_mask = self.detect_red_elements(image, yellow_mask)
        text_mask = self.detect_text_enhanced(image, yellow_mask)
        
        score_map = np.zeros((height, width), dtype=np.float32)
        score_map += yellow_mask.astype(np.float32) * 0.4
        score_map += blue_mask.astype(np.float32) * 0.2
        score_map += red_mask.astype(np.float32) * 0.2
        score_map += text_mask.astype(np.float32) * 0.2
        
        if score_map.max() > 0:
            score_map = (score_map / score_map.max() * 255)
        
        candidates_masks = []
        thresholds = [100, 120, 150]
        
        for thresh in thresholds:
            _, candidate_mask = cv2.threshold(score_map.astype(np.uint8), 
                                            thresh, 255, cv2.THRESH_BINARY)
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
                
                if area > 500:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    if (0.3 <= aspect_ratio <= 2.5 and 
                        w >= 30 and h >= 30 and 
                        w <= 800 and h <= 800):
                        
                        roi_score = score_map[y:y+h, x:x+w]
                        avg_score = np.mean(roi_score) if roi_score.size > 0 else 0
                        
                        roi_image = image[y:y+h, x:x+w]
                        color_confidence = self.verify_school_zone_colors_enhanced(roi_image)
                        
                        final_confidence = (
                            (avg_score / 255.0) * 0.6 + 
                            color_confidence * 0.4
                        )
                        
                        all_candidates.append({
                            'bbox': (x, y, x + w, y + h),
                            'area': area,
                            'confidence': final_confidence
                        })
        
        # 중복 제거 및 정렬
        final_candidates = self.remove_duplicate_candidates(all_candidates)
        final_candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        return final_candidates
    
    def remove_duplicate_candidates(self, candidates, iou_threshold=0.3):
        """IoU 기반 중복 제거"""
        if not candidates:
            return []
        
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        keep = []
        
        for candidate in candidates:
            should_keep = True
            x1, y1, x2, y2 = candidate['bbox']
            
            for kept in keep:
                kx1, ky1, kx2, ky2 = kept['bbox']
                
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
        
        color_score = 0.0
        
        if yellow_ratio > 0.2:
            color_score += 0.4
        elif yellow_ratio > 0.1:
            color_score += 0.2
        
        if blue_ratio > 0.02:
            color_score += 0.3
        if red_ratio > 0.01:
            color_score += 0.3
        
        return min(1.0, color_score)
    
    # ========== 데이터 증강 관련 메서드들 ==========
    def add_dust_storm(self, image, intensity='medium'):
        """황사 효과 추가"""
        height, width = image.shape[:2]
        
        if intensity == 'light':
            alpha = 0.5
            dust_color = (30, 200, 255)
        elif intensity == 'medium':
            alpha = 0.7
            dust_color = (40, 180, 255)
        else:  # heavy
            alpha = 0.9
            dust_color = (50, 160, 255)
        
        dust_overlay = np.full((height, width, 3), dust_color, dtype=np.uint8)
        noise = np.random.normal(0, 30, (height, width, 3))
        dust_overlay = np.clip(dust_overlay.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        result = cv2.addWeighted(image, 1-alpha, dust_overlay, alpha, 0)
        
        return result
    
    def add_night_effect(self, image, darkness_level=0.5):
        """야간 효과 추가"""
        darkened = image.astype(np.float32) * (1 - darkness_level)
        night_tint = np.zeros_like(image, dtype=np.float32)
        night_tint[:, :, 0] = 70  # 파란색 틴트
        
        result = darkened + night_tint * darkness_level
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def add_fog(self, image, intensity='medium'):
        """안개 효과 추가"""
        height, width = image.shape[:2]
        
        if intensity == 'light':
            alpha = 0.5
            fog_density = 200
        elif intensity == 'medium':
            alpha = 0.7
            fog_density = 180
        else:  # heavy
            alpha = 0.9
            fog_density = 160
        
        fog_overlay = np.full((height, width, 3), fog_density, dtype=np.uint8)
        y, x = np.ogrid[:height, :width]
        gradient = np.sin(y * 0.01) * np.cos(x * 0.01) * 30
        fog_overlay = fog_overlay.astype(np.float32) + gradient[:, :, np.newaxis]
        fog_overlay = np.clip(fog_overlay, 0, 255).astype(np.uint8)
        result = cv2.addWeighted(image, 1-alpha, fog_overlay, alpha, 0)
        
        return result
    
    def save_original_and_augmented(self, image, base_name, output_dir):
        """원본과 증강 이미지를 같은 폴더에 저장"""
        saved_files = []
        
        # 1. 원본 이미지 저장
        original_filename = f"{base_name}_ori.jpg"
        original_path = output_dir / original_filename
        cv2.imwrite(str(original_path), image)
        saved_files.append(original_filename)
        
        # 2. 증강 이미지들 저장
        augmentations = {
            f"{base_name}_dust_light.jpg": self.add_dust_storm(image, 'light'),
            f"{base_name}_dust_medium.jpg": self.add_dust_storm(image, 'medium'),
            f"{base_name}_dust_heavy.jpg": self.add_dust_storm(image, 'heavy'),
            f"{base_name}_night.jpg": self.add_night_effect(image, 0.5),
            f"{base_name}_fog_light.jpg": self.add_fog(image, 'light'),
            f"{base_name}_fog_medium.jpg": self.add_fog(image, 'medium'),
            f"{base_name}_fog_heavy.jpg": self.add_fog(image, 'heavy'),
        }
        
        for filename, aug_image in augmentations.items():
            save_path = output_dir / filename
            cv2.imwrite(str(save_path), aug_image)
            saved_files.append(filename)
        
        return saved_files
    
    # ========== 통합 처리 메서드 ==========
    def process_single_image(self, image_path, confidence_threshold=0.25):
        """단일 이미지 처리 (크롭 + 증강)"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"이미지를 읽을 수 없습니다: {image_path}")
            return {'data_files': 0, 'target_files': 0}
        
        base_name = Path(image_path).stem
        print(f"\n처리 중: {os.path.basename(image_path)}")
        
        # 1. 원본 이미지와 증강 이미지를 data 폴더에 저장
        print("  -> 원본 + 증강 이미지를 data 폴더에 저장 중...")
        data_files = self.save_original_and_augmented(image, base_name, self.data_dir)
        data_count = len(data_files)
        
        # 2. 표지판 크롭
        print("  -> 표지판 탐지 및 크롭 중...")
        masks_list, score_map = self.combine_all_features(image)
        candidates = self.extract_enhanced_candidates(image, masks_list, score_map)
        
        filtered_candidates = [c for c in candidates if c['confidence'] >= confidence_threshold]
        
        target_count = 0
        
        if filtered_candidates:
            # 상위 3개까지 크롭
            top_candidates = filtered_candidates[:3]
            
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
                    # 크롭된 이미지의 원본 + 증강을 target 폴더에 저장
                    crop_base_name = f"{base_name}_crop_{i+1:02d}"
                    target_files = self.save_original_and_augmented(cropped, crop_base_name, self.target_dir)
                    target_count += len(target_files)
                    
                    print(f"     크롭 저장: {crop_base_name}_*.jpg (신뢰도: {candidate['confidence']:.3f})")
        else:
            print(f"  -> 표지판을 찾을 수 없습니다. (임계값: {confidence_threshold})")
        
        results = {
            'data_files': data_count,
            'target_files': target_count
        }
        
        print(f"  -> data 폴더: {data_count}개, target 폴더: {target_count}개")
        return results
    
    def process_folder(self, input_folder, confidence_threshold=0.25):
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
            'total_data_files': 0,
            'total_target_files': 0,
            'failed': [],
            'success': []
        }
        
        print("=== 통합 폴더 구조 처리 시작 ===")
        print(f"총 {len(image_files)}개의 이미지를 처리합니다...")
        print(f"신뢰도 임계값: {confidence_threshold}")
        print(f"저장 구조:")
        print(f"  - {self.data_dir}/: 원본 + 증강 이미지")
        print(f"  - {self.target_dir}/: 크롭된 표지판 + 증강")
        print()
        
        for image_file in image_files:
            try:
                file_results = self.process_single_image(str(image_file), confidence_threshold)
                
                results['processed'] += 1
                results['total_data_files'] += file_results['data_files']
                results['total_target_files'] += file_results['target_files']
                
                if file_results['target_files'] > 0:
                    results['success'].append(str(image_file))
                else:
                    results['failed'].append(str(image_file))
                    
            except Exception as e:
                print(f"처리 실패 {image_file}: {e}")
                results['failed'].append(str(image_file))
        
        return results

def main():
    print("=== 통합 폴더 구조 어린이보호구역 표지판 처리 시스템 ===")
    print("기능:")
    print("1. 원본 이미지 + 증강 → data/ 폴더")
    print("2. 크롭된 표지판 + 증강 → target/ 폴더")
    print("3. 모든 파일이 같은 폴더에 저장됨")
    print()
    
    # 입력 설정
    input_folder = input("이미지 폴더 경로 (Enter = ./images): ").strip()
    if not input_folder:
        input_folder = "./images"
    
    data_folder = input("원본 데이터 저장 폴더 (Enter = data): ").strip()
    if not data_folder:
        data_folder = "data"
    
    target_folder = input("크롭 데이터 저장 폴더 (Enter = target): ").strip()
    if not target_folder:
        target_folder = "target"
    
    confidence = input("신뢰도 임계값 (0.0-1.0, Enter = 0.25): ").strip()
    try:
        confidence = float(confidence) if confidence else 0.25
    except:
        confidence = 0.25
    
    # 처리 시작
    processor = UnifiedSchoolZoneProcessor(data_folder, target_folder)
    
    if os.path.isfile(input_folder):
        print(f"단일 파일 처리: {input_folder}")
        file_results = processor.process_single_image(input_folder, confidence)
        
        print(f"\n완료!")
        print(f"data 폴더 파일: {file_results['data_files']}개")
        print(f"target 폴더 파일: {file_results['target_files']}개")
    else:
        print(f"폴더 처리: {input_folder}")
        results = processor.process_folder(input_folder, confidence)
        
        print("\n=== 최종 처리 결과 ===")
        print(f"처리된 이미지: {results['processed']}개")
        print(f"data 폴더 파일: {results['total_data_files']}개")
        print(f"target 폴더 파일: {results['total_target_files']}개")
        print(f"성공한 이미지: {len(results['success'])}개")
        print(f"실패한 이미지: {len(results['failed'])}개")
        
        print(f"\n총 생성된 파일: {results['total_data_files'] + results['total_target_files']}개")
        
        if results['failed']:
            print("\n표지판을 찾지 못한 파일들:")
            for failed_file in results['failed'][:5]:
                print(f"  - {os.path.basename(failed_file)}")
            if len(results['failed']) > 5:
                print(f"  ... 외 {len(results['failed']) - 5}개")
    
    print(f"\n파일 저장 위치:")
    print(f"- 원본 + 증강: {data_folder}/")
    print(f"- 크롭 + 증강: {target_folder}/")

if __name__ == "__main__":
    print("필요한 패키지: pip install opencv-python numpy")
    print()
    main()