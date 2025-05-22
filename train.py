# 수정된 통합 폴더 구조용 YOLOv8 학습 시스템 (에러 해결)
import os
import yaml
import cv2
import numpy as np
from pathlib import Path
import json
import shutil
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import glob
import random

class FixedUnifiedYOLOTrainer:
    def __init__(self, dataset_dir="yolo_dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / "images"
        self.labels_dir = self.dataset_dir / "labels"
        
        # 데이터셋 구조 생성
        for split in ['train', 'val', 'test']:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)
        
        print("수정된 YOLOv8 학습 시스템 초기화 완료!")
        
    def prepare_dataset_from_unified_folders(self, data_dir="data", target_dir="target", 
                                           train_ratio=0.8, val_ratio=0.15):
        """통합 폴더 구조에서 YOLO 데이터셋 준비 (에러 수정)"""
        print("통합 폴더 구조에서 YOLO 데이터셋 준비 중...")
        
        data_path = Path(data_dir)
        target_path = Path(target_dir)
        
        if not data_path.exists() or not target_path.exists():
            print(f"data 또는 target 폴더가 존재하지 않습니다!")
            print(f"data: {data_path.exists()}, target: {target_path.exists()}")
            return False
        
        # 1. 크롭된 표지판 이미지 수집 (positive samples만 사용)
        print("크롭된 표지판 이미지 수집 중...")
        target_images = list(target_path.glob("*.jpg")) + list(target_path.glob("*.png"))
        print(f"크롭된 표지판 이미지: {len(target_images)}개")
        
        if len(target_images) == 0:
            print("크롭된 표지판 이미지가 없습니다!")
            return False
        
        # 2. 이미지 필터링 (너무 작은 이미지 제외)
        valid_images = []
        for img_path in target_images:
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w = img.shape[:2]
                    if h >= 32 and w >= 32:  # 최소 크기 체크
                        valid_images.append(img_path)
            except:
                continue
        
        print(f"유효한 이미지: {len(valid_images)}개")
        
        if len(valid_images) < 10:
            print("유효한 이미지가 너무 적습니다! (최소 10개 필요)")
            return False
        
        # 3. 데이터셋 분할
        random.shuffle(valid_images)
        n_train = int(len(valid_images) * train_ratio)
        n_val = int(len(valid_images) * val_ratio)
        
        train_images = valid_images[:n_train]
        val_images = valid_images[n_train:n_train+n_val]
        test_images = valid_images[n_train+n_val:]
        
        print(f"데이터 분할: Train {len(train_images)}, Val {len(val_images)}, Test {len(test_images)}")
        
        # 4. 각 분할에 데이터 복사 및 라벨 생성
        for split, images in [('train', train_images), ('val', val_images), ('test', test_images)]:
            if len(images) == 0:
                continue
                
            print(f"\n{split} 데이터셋 처리 중... ({len(images)}개)")
            
            for i, img_path in enumerate(images):
                try:
                    # 이미지 로드
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    h, w = img.shape[:2]
                    
                    # 이미지 크기 정규화 (640x640으로 리사이즈)
                    target_size = 640
                    scale = min(target_size / w, target_size / h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    
                    # 리사이즈
                    img_resized = cv2.resize(img, (new_w, new_h))
                    
                    # 패딩 추가 (정사각형으로 만들기)
                    pad_w = (target_size - new_w) // 2
                    pad_h = (target_size - new_h) // 2
                    
                    img_padded = cv2.copyMakeBorder(
                        img_resized, pad_h, target_size - new_h - pad_h, 
                        pad_w, target_size - new_w - pad_w, 
                        cv2.BORDER_CONSTANT, value=(114, 114, 114)
                    )
                    
                    # 새 파일명 생성
                    new_img_name = f"sign_{split}_{i:04d}.jpg"
                    new_img_path = self.images_dir / split / new_img_name
                    cv2.imwrite(str(new_img_path), img_padded)
                    
                    # 라벨 파일 생성 (정규화된 좌표)
                    # 패딩을 고려한 바운딩 박스 계산
                    bbox_x_center = (pad_w + new_w / 2) / target_size
                    bbox_y_center = (pad_h + new_h / 2) / target_size
                    bbox_width = new_w / target_size
                    bbox_height = new_h / target_size
                    
                    # 좌표 클리핑 (0-1 범위)
                    bbox_x_center = max(0, min(1, bbox_x_center))
                    bbox_y_center = max(0, min(1, bbox_y_center))
                    bbox_width = max(0, min(1, bbox_width))
                    bbox_height = max(0, min(1, bbox_height))
                    
                    label_content = f"0 {bbox_x_center:.6f} {bbox_y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n"
                    
                    label_path = self.labels_dir / split / f"sign_{split}_{i:04d}.txt"
                    with open(label_path, 'w') as f:
                        f.write(label_content)
                    
                    if (i + 1) % 20 == 0:
                        print(f"  {i + 1}/{len(images)} 완료")
                        
                except Exception as e:
                    print(f"  이미지 처리 오류 {img_path}: {e}")
                    continue
        
        # 5. 최종 검증
        total_train = len(list((self.images_dir / 'train').glob('*.jpg')))
        total_val = len(list((self.images_dir / 'val').glob('*.jpg')))
        total_test = len(list((self.images_dir / 'test').glob('*.jpg')))
        
        print(f"\n✅ 데이터셋 준비 완료!")
        print(f"최종 데이터: Train {total_train}, Val {total_val}, Test {total_test}")
        
        if total_train < 5 or total_val < 2:
            print("⚠️ 경고: 데이터가 너무 적습니다. 성능이 좋지 않을 수 있습니다.")
        
        return True
    
    def create_yaml_config(self):
        """YOLO 학습용 YAML 설정 파일 생성"""
        config = {
            'path': str(self.dataset_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 1,  # number of classes
            'names': ['school_zone_sign']
        }
        
        yaml_path = self.dataset_dir / "school_zone_config.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"YAML 설정 파일 생성: {yaml_path}")
        return yaml_path
    
    def train_model(self, yaml_path, model_size='n', epochs=50, imgsz=640, batch_size=8):
        """YOLOv8 모델 학습 (에러 방지 설정)"""
        print(f"\nYOLOv8{model_size} 모델 학습 시작...")
        print(f"Epochs: {epochs}, Image size: {imgsz}, Batch size: {batch_size}")
        
        # GPU 사용 가능 여부 확인
        device = 'cpu'  # 안정성을 위해 CPU 강제 사용
        print(f"사용 디바이스: {device}")
        
        # 모델 초기화
        model = YOLO(f'yolov8{model_size}.pt')
        
        # 에러 방지를 위한 안전한 학습 설정
        try:
            results = model.train(
                data=str(yaml_path),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size,
                name='school_zone_detector_fixed',
                project='runs/detect',
                save=True,
                patience=20,  # Early stopping
                device=device,
                workers=0,      # 멀티프로세싱 비활성화
                augment=False,  # 증강 비활성화 (albumentations 에러 방지)
                mixup=0.0,      # Mixup 비활성화
                copy_paste=0.0, # Copy-paste 비활성화
                cache=False,    # 캐시 비활성화
                amp=False,      # AMP 비활성화 (CPU에서는 불필요)
                verbose=True,
                plots=True,
                val=True
            )
            
            print("✅ 학습 완료!")
            return model, results
            
        except Exception as e:
            print(f"❌ 학습 중 오류 발생: {e}")
            print("🔧 더 안전한 설정으로 재시도...")
            
            # 더 안전한 설정으로 재시도
            try:
                results = model.train(
                    data=str(yaml_path),
                    epochs=min(epochs, 30),  # 에포크 줄이기
                    imgsz=320,               # 이미지 크기 줄이기
                    batch=4,                 # 배치 크기 줄이기
                    name='school_zone_detector_safe',
                    project='runs/detect',
                    save=True,
                    patience=10,
                    device='cpu',
                    workers=0,
                    augment=False,
                    cache=False,
                    amp=False,
                    verbose=False,  # 로그 줄이기
                    plots=False
                )
                
                print("✅ 안전 모드 학습 완료!")
                return model, results
                
            except Exception as e2:
                print(f"❌ 안전 모드도 실패: {e2}")
                return None, None
    
    def validate_model(self, model_path, yaml_path):
        """모델 검증"""
        try:
            print("모델 검증 중...")
            model = YOLO(model_path)
            
            # 검증 실행
            metrics = model.val(data=str(yaml_path), device='cpu', workers=0)
            
            print(f"✅ 검증 완료!")
            if hasattr(metrics, 'box'):
                print(f"mAP50: {metrics.box.map50:.3f}")
                print(f"mAP50-95: {metrics.box.map:.3f}")
                print(f"Precision: {metrics.box.mp:.3f}")
                print(f"Recall: {metrics.box.mr:.3f}")
            
            return metrics
            
        except Exception as e:
            print(f"❌ 검증 중 오류: {e}")
            return None
    
    def export_model(self, model_path, formats=['onnx']):
        """모델을 다른 형식으로 내보내기"""
        try:
            model = YOLO(model_path)
            
            for format_type in formats:
                print(f"모델을 {format_type} 형식으로 내보내는 중...")
                try:
                    model.export(format=format_type, imgsz=640, device='cpu')
                    print(f"✅ {format_type} 모델 내보내기 완료!")
                except Exception as e:
                    print(f"❌ {format_type} 내보내기 실패: {e}")
                    
        except Exception as e:
            print(f"❌ 모델 내보내기 오류: {e}")
    
    def test_model_quick(self, model_path, test_dir, num_samples=3):
        """빠른 모델 테스트"""
        try:
            print("빠른 모델 테스트 중...")
            
            model = YOLO(model_path)
            test_path = Path(test_dir)
            
            if not test_path.exists():
                print(f"테스트 폴더가 없습니다: {test_dir}")
                return
            
            # 테스트 이미지들 가져오기
            test_images = list(test_path.glob("*.jpg")) + list(test_path.glob("*.png"))
            if len(test_images) == 0:
                print("테스트 이미지가 없습니다!")
                return
            
            # 랜덤하게 샘플 선택
            sample_images = random.sample(test_images, min(num_samples, len(test_images)))
            
            detection_count = 0
            
            for i, img_path in enumerate(sample_images):
                print(f"테스트 {i+1}/{len(sample_images)}: {img_path.name}")
                
                try:
                    # 추론
                    results = model(str(img_path), conf=0.3, device='cpu', verbose=False)
                    
                    # 결과 확인
                    if results[0].boxes is not None and len(results[0].boxes) > 0:
                        for box in results[0].boxes:
                            conf = box.conf[0].cpu().numpy()
                            print(f"  ✅ 표지판 탐지됨! 신뢰도: {conf:.3f}")
                            detection_count += 1
                    else:
                        print("  ❌ 표지판 탐지되지 않음")
                        
                except Exception as e:
                    print(f"  ❌ 테스트 오류: {e}")
            
            print(f"\n📊 테스트 결과: {detection_count}개 탐지")
            
        except Exception as e:
            print(f"❌ 테스트 오류: {e}")

def main_training():
    """수정된 학습 메인 함수"""
    print("=" * 60)
    print("🚸 수정된 YOLOv8 어린이보호구역 표지판 학습 시스템")
    print("=" * 60)
    
    # 설정
    trainer = FixedUnifiedYOLOTrainer("yolo_dataset")
    
    # 데이터 경로 입력
    data_dir = input("data 폴더 경로 (Enter = data): ").strip()
    if not data_dir:
        data_dir = "data"
    
    target_dir = input("target 폴더 경로 (Enter = target): ").strip()
    if not target_dir:
        target_dir = "target"
    
    # 학습 설정 (안전한 기본값)
    model_size = input("YOLOv8 모델 크기 (n/s, Enter = n): ").strip()
    if not model_size or model_size not in ['n', 's', 'm', 'l', 'x']:
        model_size = 'n'
    
    epochs = input("학습 에포크 수 (Enter = 50): ").strip()
    try:
        epochs = int(epochs) if epochs else 50
        epochs = min(epochs, 100)  # 최대 100으로 제한
    except:
        epochs = 50
    
    batch_size = input("배치 크기 (Enter = 8): ").strip()
    try:
        batch_size = int(batch_size) if batch_size else 8
        batch_size = min(batch_size, 16)  # 최대 16으로 제한
    except:
        batch_size = 8
    
    print(f"\n🔧 학습 설정:")
    print(f"  모델: YOLOv8{model_size}")
    print(f"  에포크: {epochs}")
    print(f"  배치 크기: {batch_size}")
    print(f"  디바이스: CPU (안정성)")
    
    # 데이터셋 준비
    print("\n" + "="*50)
    success = trainer.prepare_dataset_from_unified_folders(data_dir, target_dir)
    if not success:
        print("❌ 데이터셋 준비 실패!")
        return
    
    # YAML 설정 파일 생성
    yaml_path = trainer.create_yaml_config()
    
    # 모델 학습
    print("\n" + "="*50)
    model, results = trainer.train_model(yaml_path, model_size, epochs, batch_size=batch_size)
    
    if model is None:
        print("❌ 학습 실패!")
        return
    
    # 최적 모델 경로 찾기
    best_model_paths = list(Path('runs/detect').glob('*/weights/best.pt'))
    if best_model_paths:
        best_model_path = str(sorted(best_model_paths)[-1])  # 가장 최근 것
        print(f"✅ 최적 모델: {best_model_path}")
        
        # 모델 검증
        print("\n" + "="*50)
        trainer.validate_model(best_model_path, yaml_path)
        
        # 빠른 테스트
        print("\n" + "="*50)
        trainer.test_model_quick(best_model_path, target_dir, 3)
        
        # ONNX 내보내기
        print("\n" + "="*50)
        trainer.export_model(best_model_path, ['onnx'])
        
        print(f"\n" + "="*50)
        print("✅ 모든 작업 완료!")
        print(f"📁 모델 저장 위치: {best_model_path}")
        print("🎥 실시간 탐지를 위해 다음 명령을 실행하세요:")
        print(f"python oak_realtime_detector.py --model {best_model_path}")
    else:
        print("❌ 모델 파일을 찾을 수 없습니다!")

if __name__ == "__main__":
    print("필요한 패키지:")
    print("pip install ultralytics opencv-python")
    print("pip install 'albumentations>=1.4.0'")
    print()
    main_training()ㅍ