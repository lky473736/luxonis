# 텍스트 기반 표지판 전체 크롭퍼
# "어린이보호구역" 등의 텍스트를 찾아서 표지판 전체를 크롭

import cv2
import numpy as np
import os
from pathlib import Path

class SignBoardCropper:
    def __init__(self, output_dir="full_sign_boards"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        print("표지판 전체 크롭퍼 초기화 완료!")
    
    def detect_text_regions(self, image):
        """텍스트 영역 탐지 (어린이보호구역 관련 텍스트)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 다양한 전처리로 텍스트 탐지
        text_regions = []
        
        # 방법 1: 적응적 임계값
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        text_regions.extend(self.find_text_contours(adaptive))
        
        # 방법 2: OTSU 임계값
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_regions.extend(self.find_text_contours(otsu))
        
        # 방법 3: 대비 향상 후 임계값
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        _, enhanced_thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_regions.extend(self.find_text_contours(enhanced_thresh))
        
        return text_regions
    
    def find_text_contours(self, binary_image):
        """이진 이미지에서 텍스트 윤곽선 찾기"""
        # 텍스트 연결을 위한 모폴로지 연산
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))  # 가로 연결
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))  # 세로 연결
        
        horizontal = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel_h)
        vertical = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel_v)
        
        # 합치기
        combined = cv2.bitwise_or(horizontal, vertical)
        
        # 텍스트 라인 형성
        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
        text_lines = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_line)
        
        # 윤곽선 찾기
        contours, _ = cv2.findContours(text_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200:  # 최소 텍스트 영역
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # 텍스트 라인 특성 (가로가 긴 박스)
                if (aspect_ratio > 1.5 and 
                    w > 30 and h > 10 and 
                    w < 600 and h < 100):
                    text_regions.append((x, y, x + w, y + h))
        
        return text_regions
    
    def find_sign_board_boundary(self, image, text_regions):
        """텍스트가 포함된 표지판의 전체 경계 찾기"""
        if not text_regions:
            return []
        
        sign_candidates = []
        
        for text_bbox in text_regions:
            tx1, ty1, tx2, ty2 = text_bbox
            
            # 텍스트 주변 영역에서 표지판 경계 찾기
            sign_bbox = self.find_sign_around_text(image, text_bbox)
            
            if sign_bbox:
                sign_candidates.append(sign_bbox)
        
        # 중복 제거
        final_signs = self.merge_overlapping_signs(sign_candidates)
        
        return final_signs
    
    def find_sign_around_text(self, image, text_bbox):
        """텍스트 주변에서 표지판 전체 경계 찾기"""
        tx1, ty1, tx2, ty2 = text_bbox
        
        # 1. 색상 기반으로 표지판 영역 확장
        color_bbox = self.expand_by_color(image, text_bbox)
        
        # 2. 에지 기반으로 표지판 경계 찾기
        edge_bbox = self.find_edges_around_text(image, text_bbox)
        
        # 3. 두 방법 중 더 합리적인 것 선택
        final_bbox = self.choose_better_bbox(image, color_bbox, edge_bbox, text_bbox)
        
        return final_bbox
    
    def expand_by_color(self, image, text_bbox):
        """색상 기반으로 표지판 영역 확장"""
        tx1, ty1, tx2, ty2 = text_bbox
        
        # 텍스트 영역의 색상 분석
        text_roi = image[ty1:ty2, tx1:tx2]
        if text_roi.size == 0:
            return None
        
        # 주요 색상 추출 (노란색, 흰색, 파란색 등)
        hsv_roi = cv2.cvtColor(text_roi, cv2.COLOR_BGR2HSV)
        
        # 어린이보호구역 표지판 색상들
        color_ranges = [
            # 노란색 (배경)
            (np.array([15, 50, 100]), np.array([35, 255, 255])),
            # 흰색 (텍스트)
            (np.array([0, 0, 180]), np.array([180, 30, 255])),
            # 파란색 (삼각형)
            (np.array([100, 50, 50]), np.array([130, 255, 255]))
        ]
        
        # 전체 이미지에서 비슷한 색상 영역 찾기
        hsv_full = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        combined_mask = np.zeros(hsv_full.shape[:2], dtype=np.uint8)
        
        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv_full, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # 텍스트 영역 주변의 색상 영역 찾기
        # 텍스트를 중심으로 확장 범위 설정
        expand_factor = 3
        search_x1 = max(0, tx1 - (tx2 - tx1) * expand_factor // 2)
        search_y1 = max(0, ty1 - (ty2 - ty1) * expand_factor // 2)
        search_x2 = min(image.shape[1], tx2 + (tx2 - tx1) * expand_factor // 2)
        search_y2 = min(image.shape[0], ty2 + (ty2 - ty1) * expand_factor // 2)
        
        # 검색 영역에서 색상 마스크 적용
        search_mask = combined_mask[search_y1:search_y2, search_x1:search_x2]
        
        # 가장 큰 연결된 영역 찾기
        kernel = np.ones((5, 5), np.uint8)
        search_mask = cv2.morphologyEx(search_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(search_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 텍스트가 포함된 가장 큰 영역 찾기
            for contour in sorted(contours, key=cv2.contourArea, reverse=True):
                x, y, w, h = cv2.boundingRect(contour)
                
                # 실제 좌표로 변환
                actual_x1 = search_x1 + x
                actual_y1 = search_y1 + y
                actual_x2 = search_x1 + x + w
                actual_y2 = search_y1 + y + h
                
                # 텍스트가 이 영역에 포함되는지 확인
                if (actual_x1 <= tx1 and actual_y1 <= ty1 and 
                    actual_x2 >= tx2 and actual_y2 >= ty2):
                    return (actual_x1, actual_y1, actual_x2, actual_y2)
        
        return None
    
    def find_edges_around_text(self, image, text_bbox):
        """에지 기반으로 표지판 경계 찾기"""
        tx1, ty1, tx2, ty2 = text_bbox
        
        # 텍스트 주변 확장 영역 설정
        expand_factor = 2
        search_x1 = max(0, tx1 - (tx2 - tx1) * expand_factor // 2)
        search_y1 = max(0, ty1 - (ty2 - ty1) * expand_factor // 2)
        search_x2 = min(image.shape[1], tx2 + (tx2 - tx1) * expand_factor // 2)
        search_y2 = min(image.shape[0], ty2 + (ty2 - ty1) * expand_factor // 2)
        
        # 검색 영역 추출
        search_roi = image[search_y1:search_y2, search_x1:search_x2]
        gray_roi = cv2.cvtColor(search_roi, cv2.COLOR_BGR2GRAY)
        
        # 에지 탐지
        edges = cv2.Canny(gray_roi, 50, 150)
        
        # 직사각형 윤곽선 찾기
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            # 윤곽선 근사
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) >= 4:  # 사각형에 가까운 모양
                x, y, w, h = cv2.boundingRect(contour)
                
                # 실제 좌표로 변환
                actual_x1 = search_x1 + x
                actual_y1 = search_y1 + y
                actual_x2 = search_x1 + x + w
                actual_y2 = search_y1 + y + h
                
                # 텍스트가 포함되고 합리적인 크기인지 확인
                if (actual_x1 <= tx1 and actual_y1 <= ty1 and 
                    actual_x2 >= tx2 and actual_y2 >= ty2 and
                    w > 50 and h > 30 and w < 800 and h < 600):
                    return (actual_x1, actual_y1, actual_x2, actual_y2)
        
        return None
    
    def choose_better_bbox(self, image, color_bbox, edge_bbox, text_bbox):
        """두 바운딩박스 중 더 적합한 것 선택"""
        tx1, ty1, tx2, ty2 = text_bbox
        
        candidates = []
        if color_bbox:
            candidates.append(('color', color_bbox))
        if edge_bbox:
            candidates.append(('edge', edge_bbox))
        
        if not candidates:
            # 기본값: 텍스트 영역을 2.5배 확장
            text_w = tx2 - tx1
            text_h = ty2 - ty1
            center_x = (tx1 + tx2) // 2
            center_y = (ty1 + ty2) // 2
            
            new_w = int(text_w * 2.5)
            new_h = int(text_h * 2.5)
            
            default_x1 = max(0, center_x - new_w // 2)
            default_y1 = max(0, center_y - new_h // 2)
            default_x2 = min(image.shape[1], center_x + new_w // 2)
            default_y2 = min(image.shape[0], center_y + new_h // 2)
            
            return (default_x1, default_y1, default_x2, default_y2)
        
        # 가장 합리적인 크기의 bbox 선택
        best_bbox = None
        best_score = 0
        
        for method, bbox in candidates:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            area = width * height
            aspect_ratio = width / height
            
            # 점수 계산 (표지판다운 비율과 크기)
            score = 0
            
            # 크기 점수
            if 2000 <= area <= 200000:
                score += 0.4
            
            # 비율 점수 (표지판은 보통 가로가 좀 더 김)
            if 0.8 <= aspect_ratio <= 3.0:
                score += 0.4
            
            # 텍스트 위치 점수 (텍스트가 중앙이나 상단에 위치)
            text_center_x = (tx1 + tx2) / 2
            text_center_y = (ty1 + ty2) / 2
            bbox_center_x = (x1 + x2) / 2
            bbox_center_y = (y1 + y2) / 2
            
            x_diff = abs(text_center_x - bbox_center_x) / width
            y_ratio = (text_center_y - y1) / height
            
            if x_diff < 0.3 and 0.2 <= y_ratio <= 0.8:
                score += 0.2
            
            if score > best_score:
                best_score = score
                best_bbox = bbox
        
        return best_bbox if best_bbox else candidates[0][1]
    
    def merge_overlapping_signs(self, sign_candidates):
        """겹치는 표지판 영역들을 병합"""
        if not sign_candidates:
            return []
        
        # IoU가 높은 박스들을 병합
        merged = []
        used = set()
        
        for i, bbox1 in enumerate(sign_candidates):
            if i in used:
                continue
            
            # 현재 박스와 겹치는 모든 박스 찾기
            group = [bbox1]
            used.add(i)
            
            for j, bbox2 in enumerate(sign_candidates):
                if j in used:
                    continue
                
                iou = self.calculate_iou(bbox1, bbox2)
                if iou > 0.3:  # 30% 이상 겹치면 같은 표지판으로 간주
                    group.append(bbox2)
                    used.add(j)
            
            # 그룹의 모든 박스를 포함하는 최소 박스 생성
            if group:
                x1s, y1s, x2s, y2s = zip(*group)
                merged_bbox = (min(x1s), min(y1s), max(x2s), max(y2s))
                merged.append(merged_bbox)
        
        return merged
    
    def calculate_iou(self, bbox1, bbox2):
        """IoU 계산"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 교집합
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
    
    def crop_and_save_signs(self, image_path):
        """표지판 전체를 탐지하고 크롭하여 저장"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"이미지를 읽을 수 없습니다: {image_path}")
            return 0
        
        print(f"처리 중: {os.path.basename(image_path)}")
        
        # 1. 텍스트 영역 탐지
        text_regions = self.detect_text_regions(image)
        
        if not text_regions:
            print(f"  -> 텍스트를 찾을 수 없습니다.")
            return 0
        
        print(f"  -> {len(text_regions)}개의 텍스트 영역 발견")
        
        # 2. 표지판 전체 경계 찾기
        sign_boundaries = self.find_sign_board_boundary(image, text_regions)
        
        if not sign_boundaries:
            print(f"  -> 표지판 경계를 찾을 수 없습니다.")
            return 0
        
        print(f"  -> {len(sign_boundaries)}개의 표지판 발견")
        
        # 3. 크롭 및 저장
        base_name = Path(image_path).stem
        saved_count = 0
        
        for i, bbox in enumerate(sign_boundaries):
            x1, y1, x2, y2 = bbox
            
            # 패딩 추가
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)
            
            # 크롭
            cropped = image[y1:y2, x1:x2]
            
            if cropped.size > 0 and cropped.shape[0] > 30 and cropped.shape[1] > 30:
                # 저장
                output_filename = f"{base_name}_sign_board_{i+1:02d}.jpg"
                output_path = self.output_dir / output_filename
                cv2.imwrite(str(output_path), cropped)
                saved_count += 1
                
                print(f"  -> 저장: {output_filename} (크기: {x2-x1}x{y2-y1})")
        
        return saved_count
    
    def process_folder(self, input_folder):
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
        print()
        
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
    print("=== 표지판 전체 크롭퍼 ===")
    print("어린이보호구역 텍스트를 찾아서 표지판 전체를 크롭")
    print()
    
    # 입력 설정
    input_folder = input("이미지 폴더 경로 (Enter = ./images): ").strip()
    if not input_folder:
        input_folder = "./images"
    
    output_folder = input("출력 폴더명 (Enter = full_sign_boards): ").strip()
    if not output_folder:
        output_folder = "full_sign_boards"
    
    # 크로퍼 초기화 및 실행
    cropper = SignBoardCropper(output_folder)
    
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
            print("\n처리 실패한 파일들:")
            for failed_file in results['failed'][:5]:
                print(f"  - {os.path.basename(failed_file)}")
    
    print(f"\n표지판 전체가 '{output_folder}' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    print("필요한 패키지: pip install opencv-python numpy")
    print()
    main()