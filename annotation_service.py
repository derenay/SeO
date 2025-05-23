"""
Purpose: The ImageAnnotationService is responsible for drawing visual feedback on images.
Key Idea: After objects are detected and recognized,
 this service takes the original image and the recognition results (bounding boxes, object names, confidence scores) and draws them onto the image, 
 creating the annotated output image you see. It handles font loading and text placement.
"""
import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional

from config import SystemConfig

logger = logging.getLogger(__name__)

class ImageAnnotationService:
    """
    Handles drawing annotations (bounding boxes, labels) on images.
    """
    def __init__(self, config: SystemConfig):
        self.config = config
        # TTF fontları için FreeType kullanımı daha karmaşıktır.
        # Şimdilik OpenCV'nin dahili Hershey fontlarını kullanacağız.
        # Eğer config.FONT_PATH belirtilmişse ve OpenCV FreeType ile derlenmişse,
        # cv2.freetype.createFreeType2() kullanılabilir.
        self.font_face = self.config.OPENCV_FONT_FACE
        self.font_scale_small = self.config.OPENCV_FONT_SCALE_SMALL
        self.text_thickness = self.config.OPENCV_TEXT_THICKNESS


    def _get_dynamic_font_scale(self, image_height: int) -> float:
        """Calculates a dynamic font scale based on image height."""
        # Bu bir örnek, daha iyi bir ölçekleme formülü bulunabilir
        return max(0.4, min(2.0, image_height / self.config.OPENCV_FONT_SCALE_VIDEO_DIVISOR))




    def draw_annotations_on_image(self,
                                 image_cv: np.ndarray, # Artık OpenCV imajı (BGR NumPy array)
                                 annotations: List[Dict[str, Any]],
                                 font_scale_override: Optional[float] = None
                                 ) -> np.ndarray:
        """
        Draws annotations (bounding boxes and labels) on a copy of the input OpenCV image.

        Args:
            image_cv (np.ndarray): The OpenCV Image (BGR format) to draw on.
            annotations (List[Dict[str, Any]]): A list of annotation dictionaries.
                Each dict should have:
                - 'box': [x1, y1, x2, y2] coordinates
                - 'label': Text string for the label
                - 'color': BGR tuple e.g., (255, 0, 0) for blue
            font_scale_override (Optional[float]): Specific font scale for OpenCV.

        Returns:
            np.ndarray: A new OpenCV image (BGR format) with annotations drawn.
        """
        if not annotations:
            return image_cv.copy()

        annotated_image = image_cv.copy()
        height, _ = annotated_image.shape[:2]

        font_scale_to_use = font_scale_override if font_scale_override is not None \
                            else self._get_dynamic_font_scale(height)
        
        line_thickness = max(1, int(height / self.config.ANNOTATION_BOX_THICKNESS_DIVISOR))

        for ann in annotations:
            box = ann['box']
            label = ann['label']
            # OpenCV renkleri BGR formatında olmalı
            bgr_color = ann['color'] if isinstance(ann['color'], tuple) else (255,0,0) # Default to blue

            x1, y1, x2, y2 = map(int, box)

            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), bgr_color, line_thickness)

            # Text için hazırlık
            (text_width, text_height), baseline = cv2.getTextSize(label, self.font_face,
                                                                 font_scale_to_use, self.text_thickness)
            
            # Text arka planı için pozisyon
            # Etiketi kutunun üstüne yerleştir
            label_bg_y1 = y1 - text_height - baseline - (line_thickness) # Biraz boşluk bırak
            if label_bg_y1 < 0: # Eğer metin yukarıdan taşıyorsa, kutunun içine veya altına al
                label_bg_y1 = y1 + baseline + line_thickness # Kutunun hemen altına

            label_bg_x1 = x1
            label_bg_x2 = x1 + text_width
            label_bg_y2 = label_bg_y1 + text_height + baseline

            # Arka planı çiz (metnin okunurluğunu artırmak için)
            cv2.rectangle(annotated_image, (label_bg_x1, label_bg_y1 - text_height - (baseline//2)), # y1'i düzelt
                          (label_bg_x2, label_bg_y2 - (baseline//2)), bgr_color, -1) # -1 for filled

            # Metni yaz
            cv2.putText(annotated_image, label, (x1, label_bg_y1 + text_height), # Metin pozisyonu
                        self.font_face, font_scale_to_use, (0,0,0), # Siyah metin
                        self.text_thickness, cv2.LINE_AA)

        return annotated_image