import cv2
import numpy as np
import os
from pathlib import Path
import random

class SchoolZoneAugmentation :
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def add_dust_storm(self, image, intensity='medium'):
        height, width = image.shape[:2]
        
        if intensity == 'light':
            alpha = 0.5
            dust_color = (30, 200, 255)
        elif intensity == 'medium':
            alpha = 0.7
            dust_color = (40, 180, 255)
        else:
            alpha = 0.9
            dust_color = (50, 160, 255)
        
        dust_overlay = np.full((height, width, 3), dust_color, dtype=np.uint8)
        noise = np.random.normal(0, 30, (height, width, 3))
        dust_overlay = np.clip(dust_overlay.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        result = cv2.addWeighted(image, 1-alpha, dust_overlay, alpha, 0)
        
        return result
    
    def add_night_effect(self, image, darkness_level=0.9):
        darkened = image.astype(np.float32) * (1 - darkness_level)
        night_tint = np.zeros_like(image, dtype=np.float32)
        night_tint[:, :, 0] = 70
        
        result = darkened + night_tint * darkness_level
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def add_fog(self, image, intensity='medium'):
        height, width = image.shape[:2]
        
        if intensity == 'light':
            alpha = 0.5
            fog_density = 200
        elif intensity == 'medium':
            alpha = 0.7
            fog_density = 180
        else:
            alpha = 0.9
            fog_density = 160
        
        fog_overlay = np.full((height, width, 3), fog_density, dtype=np.uint8)
        y, x = np.ogrid[:height, :width]
        gradient = np.sin(y * 0.01) * np.cos(x * 0.01) * 30
        fog_overlay = fog_overlay.astype(np.float32) + gradient[:, :, np.newaxis]
        fog_overlay = np.clip(fog_overlay, 0, 255).astype(np.uint8)
        result = cv2.addWeighted(image, 1-alpha, fog_overlay, alpha, 0)
        
        return result
    
    def process_single_image(self, image_path):
        image = cv2.imread(str(image_path))
        
        base_name = image_path.stem
        extension = image_path.suffix
        
        dust_light = self.add_dust_storm(image, 'light')
        dust_medium = self.add_dust_storm(image, 'medium')
        dust_heavy = self.add_dust_storm(image, 'heavy')
        night_image = self.add_night_effect(image, darkness_level=0.5)
        fog_light = self.add_fog(image, 'light')
        fog_medium = self.add_fog(image, 'medium')
        fog_heavy = self.add_fog(image, 'heavy')
        
        augmented_images = {
            f"{base_name}_dust_light{extension}": dust_light,
            f"{base_name}_dust_medium{extension}": dust_medium,
            f"{base_name}_dust_heavy{extension}": dust_heavy,
            f"{base_name}_night{extension}": night_image,
            f"{base_name}_fog_light{extension}": fog_light,
            f"{base_name}_fog_medium{extension}": fog_medium,
            f"{base_name}_fog_heavy{extension}": fog_heavy,
        }
        
        for filename, aug_image in augmented_images.items():
            save_path = self.output_dir / filename
            cv2.imwrite(str(save_path), aug_image)
            print(f"저장완료: {save_path}")
    
    def process_all_images(self):
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [
            f for f in self.input_dir.iterdir() 
            if f.suffix.lower() in image_extensions and f.is_file()
        ]
        
        print(f"{len(image_files)}개의 이미지를 처리...")
        
        for i, image_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] 처리중: {image_file.name}")
            self.process_single_image(image_file)
        
        print(f"\n{len(image_files) * 7}개의 증강 이미지가 생성")

if __name__ == "__main__":
    input_folder = "original_images"
    output_folder = "augmented_images"
    
    augmentor = SchoolZoneAugmentation(input_folder, output_folder)
    augmentor.process_all_images()
    
    print ("=== 데이터 증강 완료 ===")
    print ("- 황사 효과: light, medium, heavy")
    print ("- 밤 효과: 어둡게 처리")
    print ("- 안개 효과: light, medium, heavy")
    print ("원본 1장당 총 7장의 증강 이미지가 생성됩니다.")