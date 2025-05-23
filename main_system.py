"""
Purpose: This file contains the main ObjectRecognitionSystem class, which now acts as an orchestrator or a conductor.
 It ties all the other components (Config, ModelManager, ObjectEmbedder, FaissDBManager, ImageAnnotationService) together to perform the complete object recognition pipeline.
Key Idea: You'll primarily interact with this class. It provides high-level methods like:

    Enrolling new objects (from a folder of images or by providing images directly).
    Deleting enrolled objects.
    Recognizing objects in a single image and saving the result.
    Recognizing objects in a video and saving the result.

The if __name__ == "__main__": block at the end of this file demonstrates how to instantiate and use the ObjectRecognitionSystem.
"""
import os
import json
import cv2 
import time
import datetime
import shutil
import pandas as pd
import numpy as np
import gc
import torch 
import logging
from typing import List, Dict, Tuple, Any, Optional

from config import SystemConfig, default_config
from model_manager import ModelManager
from embedding_service import ObjectEmbedder
from faiss_manager import FaissDBManager
from annotation_service import ImageAnnotationService

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def get_output_paths(config: SystemConfig, input_path: str, processing_date_str: str, is_video: bool = False) -> Tuple[str, str]:
    """Generates output paths for processed media and JSON results."""
    safe_processing_date_str = processing_date_str.replace(":", "-").replace(" ", "_")
    base_output_dir = os.path.join(config.OUTPUT_BASE_DIR, safe_processing_date_str)
    os.makedirs(base_output_dir, exist_ok=True)

    original_filename = os.path.basename(input_path)
    name, ext = os.path.splitext(original_filename)

    processed_media_filename = f"{name}_processed{'.mp4' if is_video else ext}" 
    json_filename = f"{name}_results.json"

    output_media_path = os.path.join(base_output_dir, processed_media_filename)
    output_json_path = os.path.join(base_output_dir, json_filename)

    return output_media_path, output_json_path


class ObjectRecognitionSystem:
    """
    Orchestrates object recognition using YOLO, ResNet embeddings, and FAISS.
    This class now delegates tasks to specialized managers.
    """
    def __init__(self, config: SystemConfig = default_config):
        logger.info("Initializing Object Recognition System...")
        self.config = config
        self.model_manager = ModelManager(config)
        self.embedder = ObjectEmbedder(self.model_manager, config)
        self.faiss_db = FaissDBManager(config)
        self.annotator = ImageAnnotationService(config)
        logger.info("Object Recognition System initialized successfully.")

    def scan_and_enroll_from_filesystem(self, dataset_path: Optional[str] = None) -> Tuple[str, pd.DataFrame]:
        """Scans a directory for object images and enrolls/updates them in FAISS."""
        if not self.model_manager.resnet_ready:
            msg = "Error: Embedding model (ResNet50) not loaded."
            logger.error(msg)
            return msg, self.get_enrolled_objects_dataframe()

        current_dataset_path = dataset_path if dataset_path else self.config.DEFAULT_DATASET_BASE_DIR
        logger.info(f"Starting prototype update from filesystem: '{current_dataset_path}'")

        if not os.path.isdir(current_dataset_path):
            msg = f"Error: Dataset path '{current_dataset_path}' not found."
            logger.error(msg)
            return msg, self.get_enrolled_objects_dataframe()

        object_folders = [d for d in os.listdir(current_dataset_path)
                          if os.path.isdir(os.path.join(current_dataset_path, d))]
        if not object_folders:
            msg = "Warning: No object subfolders found in dataset directory."
            logger.warning(msg)
            return msg, self.get_enrolled_objects_dataframe()

        new_prototypes_for_faiss = {} # name: {prototype_vector, num_samples, source}
        total_folders_processed = 0

        for object_name in object_folders:
            object_dir = os.path.join(current_dataset_path, object_name)
            current_object_embeddings = []
            img_count = 0
            logger.info(f"Processing object: '{object_name}' from '{object_dir}'")

            image_files = sorted([f for f in os.listdir(object_dir)
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])

            for image_file in image_files:
                if img_count >= self.config.MAX_IMAGES_PER_OBJECT_FS:
                    logger.info(f"Reached max images ({self.config.MAX_IMAGES_PER_OBJECT_FS}) for '{object_name}'.")
                    break
                image_path = os.path.join(object_dir, image_file)
                try:
                    img_cv = cv2.imread(image_path) # OpenCV ile oku (BGR)
                    if img_cv is None:
                        logger.warning(f"Could not read image {image_path}, skipping.")
                        continue
                    embedding = self.embedder.get_object_embedding(img_cv) # Embedder OpenCV imajı alır
                    if embedding is not None:
                        current_object_embeddings.append(embedding)
                        img_count += 1
                except Exception as e:
                    logger.error(f"Error processing image '{image_path}': {e}", exc_info=True)


            if current_object_embeddings:
                prototype_vector = np.mean(current_object_embeddings, axis=0).astype('float32')
                # Embedder should return normalized vectors, but ensure here if needed
                # faiss.normalize_L2(prototype_vector.reshape(1, -1)) # Handled by embedder

                new_prototypes_for_faiss[object_name] = {
                    'prototype_vector': prototype_vector.flatten(),
                    'num_samples': len(current_object_embeddings),
                    'source': 'filesystem'
                }
                total_folders_processed += 1
                logger.info(f"Prototype for '{object_name}' created from {len(current_object_embeddings)} samples.")
        
        if not new_prototypes_for_faiss:
            msg = "No valid object prototypes found in the filesystem scan."
            logger.warning(msg)
            return msg, self.get_enrolled_objects_dataframe()

        if self.faiss_db.replace_all_prototypes(new_prototypes_for_faiss):
            status_message = f"{total_folders_processed} object(s) processed/updated in FAISS from filesystem."
            logger.info(status_message)
        else:
            status_message = "Error: Failed to update FAISS index from filesystem scan. Check logs."
            logger.error(status_message)
        
        return status_message, self.get_enrolled_objects_dataframe()

    def enroll_object_manually(self, object_name_manual: str,
                               image_cv_list: List[np.ndarray]) -> Tuple[str, pd.DataFrame]: # Artık NumPy array listesi
        if not self.model_manager.resnet_ready:
             msg = "Error: Embedding model (ResNet50) not ready."
             logger.error(msg)
             return msg, self.get_enrolled_objects_dataframe()
        if not object_name_manual.strip():
            msg = "Error: Object name cannot be empty."
            logger.error(msg)
            return msg, self.get_enrolled_objects_dataframe()
        if not image_cv_list:
            msg = "Error: At least one image must be provided for manual enrollment."
            logger.error(msg)
            return msg, self.get_enrolled_objects_dataframe()

        valid_images = [img for img in image_cv_list if isinstance(img, np.ndarray)]
        if not valid_images:
            msg = "Error: No valid OpenCV Images (NumPy arrays) provided."
            logger.error(msg)
            return msg, self.get_enrolled_objects_dataframe()

        embeddings_list = []
        for i, img_cv in enumerate(valid_images):
            embedding = self.embedder.get_object_embedding(img_cv)
            if embedding is not None:
                embeddings_list.append(embedding)
            else:
                msg = f"Error: Could not extract features from image {i+1} for '{object_name_manual}'."
                logger.error(msg)
                return msg, self.get_enrolled_objects_dataframe()

        if not embeddings_list:
            msg = f"Error: No features extracted from any provided images for '{object_name_manual}'."
            logger.error(msg)
            return msg, self.get_enrolled_objects_dataframe()

        prototype_vector = np.mean(embeddings_list, axis=0).astype('float32')
        object_name = object_name_manual.strip()
        if self.faiss_db.add_or_update_prototype(object_name, prototype_vector,
                                                 len(embeddings_list), 'manual'):
            msg = f"Object '{object_name}' enrolled/updated manually with {len(embeddings_list)} samples."
            logger.info(msg)
        else:
            msg = f"Error: Failed to enroll/update '{object_name}' in FAISS. Check logs."
            logger.error(msg)
        return msg, self.get_enrolled_objects_dataframe()

    def delete_enrolled_object(self, object_name_to_delete: str) -> Tuple[str, bool, pd.DataFrame]:
        """Deletes an enrolled object from FAISS."""
        deleted = self.faiss_db.delete_prototype(object_name_to_delete)
        if deleted:
            msg = f"Object '{object_name_to_delete}' successfully deleted from FAISS."
            logger.info(msg)
        else:
            msg = f"Error or Warning: Object '{object_name_to_delete}' not found or failed to delete. Check logs."
            # Log level might depend on whether not found is an error or expected
            logger.warning(msg) 
        return msg, deleted, self.get_enrolled_objects_dataframe()

    def get_enrolled_objects_summary(self) -> List[Dict[str, Any]]:
        """Gets a summary list of all enrolled objects."""
        return self.faiss_db.get_all_metadata_summary()

    def get_enrolled_objects_dataframe(self) -> pd.DataFrame:
        """Gets a Pandas DataFrame summarizing all enrolled objects."""
        summary_list = self.get_enrolled_objects_summary()
        if not summary_list:
            return pd.DataFrame(columns=["Nesne Adı", "Örnek Sayısı", "Kaynak", "FAISS ID"])
        df = pd.DataFrame(summary_list)
        # Rename columns to match original output if needed, or keep new names
        df = df.rename(columns={'name': 'Nesne Adı', 'num_samples': 'Örnek Sayısı',
                                'source': 'Kaynak', 'faiss_id': 'FAISS ID'})
        return df

    def _process_frame_for_objects(self, frame_cv_bgr: np.ndarray, # Giriş BGR OpenCV imajı
                                   current_distance_threshold: float
                                   ) -> Tuple[np.ndarray, str, List[Dict[str, Any]]]: # Çıkış da BGR OpenCV imajı
        if not self.model_manager.yolo_model or not self.model_manager.resnet_ready:
            msg = "Error: YOLO or ResNet model not ready for processing."
            logger.error(msg)
            return frame_cv_bgr.copy(), msg, []

        current_faiss_index, current_id_map = self.faiss_db.get_current_search_state()
        annotations_for_drawing: List[Dict[str, Any]] = []
        structured_results: List[Dict[str, Any]] = []
        recognition_info_list: List[str] = []

        # YOLO.predict BGR veya RGB NumPy array veya PIL imajı alabilir.
        # Doğrudan frame_cv_bgr (NumPy BGR) verelim.
        try:
            yolo_results = self.model_manager.yolo_model.predict(
                frame_cv_bgr, # OpenCV BGR imajı
                verbose=False,
                conf=self.config.YOLO_CONFIDENCE_THRESHOLD,
            )
        except Exception as e:
            msg = f"YOLO prediction error: {e}"
            logger.error(msg, exc_info=True)
            return frame_cv_bgr.copy(), msg, []

        for result_set in yolo_results:
            if result_set.boxes is None or len(result_set.boxes) == 0:
                continue
            for box_data in result_set.boxes:
                x1, y1, x2, y2 = map(int, box_data.xyxy[0].tolist())
                yolo_conf = float(box_data.conf[0])
                coco_class_id = int(box_data.cls[0].item())
                coco_class_name = result_set.names.get(coco_class_id, f"ID:{coco_class_id}")

                object_crop_cv = frame_cv_bgr[y1:y2, x1:x2] # OpenCV crop (BGR)
                if object_crop_cv.size == 0: # Eğer crop boşsa atla
                    logger.warning(f"Empty crop for {coco_class_name} at {x1,y1,x2,y2}. Skipping.")
                    continue
                
                query_embedding = self.embedder.get_object_embedding(object_crop_cv) # Embedder BGR alır
                
                identity = f"Unknown ({coco_class_name})"
                # OpenCV renkleri (B, G, R)
                bgr_color = (255, 0, 0) # Mavi (tespit edildi, tanınmadı)
                recognition_distance = float('inf')
                best_match_name_from_faiss = "N/A"
                is_recognized = False

                current_detection_info = {
                    'bounding_box': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                    'yolo_class': coco_class_name,
                    'yolo_confidence': round(yolo_conf, 3),
                    'embedding_generated': query_embedding is not None,
                    'recognized_object_name': None,
                    'recognition_distance': None,
                    'closest_prototype_match_in_db': None
                }

                if query_embedding is not None and current_faiss_index and current_faiss_index.ntotal > 0:
                    try:
                        distances_faiss, indices_faiss = current_faiss_index.search(
                            query_embedding.reshape(1, -1).astype('float32'),
                            k=self.config.FAISS_SEARCH_TOP_K
                        )
                        if distances_faiss is not None and indices_faiss is not None and len(indices_faiss[0]) > 0:
                            faiss_id = indices_faiss[0][0]
                            similarity_score = distances_faiss[0][0]
                            recognition_distance = 1.0 - similarity_score
                            if 0 <= faiss_id < len(current_id_map):
                                match_meta = current_id_map[faiss_id]
                                best_match_name_from_faiss = match_meta['name']
                                current_detection_info['closest_prototype_match_in_db'] = best_match_name_from_faiss
                                current_detection_info['recognition_distance'] = round(float(recognition_distance), 4)
                                print(f"{recognition_distance} |||||| {current_distance_threshold}")
                                if recognition_distance < current_distance_threshold:
                                    identity = best_match_name_from_faiss
                                    bgr_color = (0, 255, 0) # Yeşil (tanındı)
                                    is_recognized = True
                                    current_detection_info['recognized_object_name'] = identity
                                else: 
                                    bgr_color = (128, 0, 128) # Mor (embedding OK, ama uzak)
                    except Exception as e_f_search:
                        logger.error(f"FAISS search error: {e_f_search}", exc_info=True)
                elif query_embedding is None:
                    bgr_color = (0, 165, 255) # Turuncu (embedding hatası)
                    identity = f"NoFeat ({coco_class_name})"

                label_text = f"{identity}"
                if is_recognized:
                    label_text += f" ({recognition_distance:.2f})"
                elif query_embedding is not None and best_match_name_from_faiss != "N/A":
                    label_text += f" (Closest:{best_match_name_from_faiss} @{recognition_distance:.2f})"

                annotations_for_drawing.append({
                    'box': [x1, y1, x2, y2], 'label': label_text, 'color': bgr_color
                })
                structured_results.append(current_detection_info)
                if is_recognized:
                    recognition_info_list.append(f"Recognized: {identity} (Dist:{recognition_distance:.3f})")
                elif query_embedding is not None:
                     recognition_info_list.append(f"Detected: {coco_class_name} (Closest:{best_match_name_from_faiss}, Dist:{recognition_distance:.3f})")
                else:
                    recognition_info_list.append(f"NoFeat: {coco_class_name}")

        height, width = frame_cv_bgr.shape[:2]
        dynamic_font_scale = max(0.4, min(2.0, width / self.config.OPENCV_FONT_SCALE_VIDEO_DIVISOR))

        annotated_frame_cv = self.annotator.draw_annotations_on_image(
            frame_cv_bgr, annotations_for_drawing, font_scale_override=dynamic_font_scale
        )
        summary_text = "\n".join(recognition_info_list) if recognition_info_list else "No significant objects found/recognized."
        return annotated_frame_cv, summary_text, structured_results

    def recognize_image_and_save(self, input_image_path: str, processing_date_str: str,
                                 distance_threshold: Optional[float] = None) -> Tuple[Optional[str], Optional[str], str]:
        if not os.path.exists(input_image_path):
            msg = f"Error: Input image not found at '{input_image_path}'"
            logger.error(msg)
            return None, None, msg
        
        input_image_cv = cv2.imread(input_image_path) # OpenCV ile oku
        if input_image_cv is None:
            msg = f"Error: Could not open or read input image '{input_image_path}' with OpenCV."
            logger.error(msg)
            return None, None, msg

        current_threshold = float(distance_threshold if distance_threshold is not None
                                  else self.config.DEFAULT_DISTANCE_THRESHOLD)
        annotated_frame_cv, summary, structured_results = self._process_frame_for_objects(
            input_image_cv, current_threshold
        )
        output_media_path, output_json_path = get_output_paths(
            self.config, input_image_path, processing_date_str, is_video=False
        )
        try:
            cv2.imwrite(output_media_path, annotated_frame_cv) # OpenCV ile kaydet
            logger.info(f"Processed image saved to: {output_media_path}")
        except Exception as e:
            summary += f"\nWARNING: Failed to save annotated image: {e}"
            logger.error(f"Failed to save annotated image to '{output_media_path}': {e}", exc_info=True)
            output_media_path = None
        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(structured_results, f, indent=4, ensure_ascii=False)
            logger.info(f"JSON results saved to: {output_json_path}")
        except Exception as e:
            summary += f"\nWARNING: Failed to save JSON results: {e}"
            logger.error(f"Failed to save JSON results to '{output_json_path}': {e}", exc_info=True)
            output_json_path = None
        return output_media_path, output_json_path, summary

    def recognize_video_and_save(self, video_path_input: str, processing_date_str: str,
                                 distance_threshold: Optional[float] = None,
                                 process_every_n_frames: Optional[int] = None
                                 ) -> Tuple[Optional[str], Optional[str], str]:
        if not os.path.exists(video_path_input):
            msg = f"Error: Video not found at '{video_path_input}'"; logger.error(msg); return None, None, msg
        cap = cv2.VideoCapture(video_path_input)
        if not cap.isOpened():
            msg = f"Error: Could not open video '{video_path_input}'"; logger.error(msg); return None, None, msg
        try:
            fps = cap.get(cv2.CAP_PROP_FPS); L_FPS = 30.0 if fps <= 0 or fps > 120 else fps
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except Exception as e:
            cap.release(); msg = f"Error reading video properties: {e}"; logger.error(msg, exc_info=True); return None, None, msg
        if w == 0 or h == 0:
            cap.release(); msg = "Error: Video dimensions are zero."; logger.error(msg); return None, None, msg

        output_media_path, output_json_path = get_output_paths(
            self.config, video_path_input, processing_date_str, is_video=True
        )
        writer = cv2.VideoWriter(output_media_path, cv2.VideoWriter_fourcc(*'mp4v'), L_FPS, (w, h))
        if not writer.isOpened():
            cap.release(); msg = f"Error: VideoWriter for '{output_media_path}'"; logger.error(msg); return None, None, msg

        frame_idx = 0; all_frames_json_results = []; summary_texts_for_video = []
        last_known_annotations_for_drawing: List[Dict[str, Any]] = []
        current_threshold = float(distance_threshold if distance_threshold is not None else self.config.DEFAULT_DISTANCE_THRESHOLD)
        _process_every_n_frames = process_every_n_frames if process_every_n_frames is not None else self.config.PROCESS_EVERY_N_FRAMES
        
        logger.info(f"Processing video '{video_path_input}' (approx. {total_f} frames). Output: {output_media_path}")
        try:
            while True:
                ret, frame_bgr = cap.read() # frame_bgr zaten OpenCV BGR formatında
                if not ret: break
                frame_idx += 1
                
                output_cv_frame_to_write: np.ndarray # Type hint
                if frame_idx % _process_every_n_frames == 1 or _process_every_n_frames == 1:
                    annotated_cv_frame, summary, structured_res = self._process_frame_for_objects(
                        frame_bgr, current_threshold
                    )
                    output_cv_frame_to_write = annotated_cv_frame
                    last_known_annotations_for_drawing = [] # _process_frame_for_objects'ten gelen çizim bilgisini kullan
                    for sr_item in structured_res:
                        label_vid = sr_item.get('recognized_object_name') or f"Unknown ({sr_item.get('yolo_class')})"
                        if sr_item.get('recognition_distance') is not None and sr_item.get('recognized_object_name'):
                             label_vid += f" ({sr_item.get('recognition_distance'):.2f})"
                        elif not sr_item.get('embedding_generated', False):
                             label_vid = f"NoFeat ({sr_item.get('yolo_class')})"
                        
                        color_vid = (0,255,0) if sr_item.get('recognized_object_name') else (255,0,0) # Yeşil, Mavi (BGR)
                        if not sr_item.get('embedding_generated', False): color_vid = (0,165,255) # Turuncu
                        elif sr_item.get('embedding_generated',True) and not sr_item.get('recognized_object_name') and sr_item.get('recognition_distance') is not None:
                            color_vid = (128,0,128) # Mor

                        last_known_annotations_for_drawing.append({
                            'box': [sr_item['bounding_box']['x1'], sr_item['bounding_box']['y1'], 
                                    sr_item['bounding_box']['x2'], sr_item['bounding_box']['y2']],
                            'label': label_vid, 'color': color_vid
                        })
                    if structured_res:
                        all_frames_json_results.append({
                            "frame_index": frame_idx, "detections": structured_res, "frame_summary": summary
                        })
                    if summary and "Error:" not in summary:
                        summary_texts_for_video.append(f"Frame {frame_idx}: {summary}")
                else:
                    dynamic_font_scale_video = max(0.4, min(2.0, w / self.config.OPENCV_FONT_SCALE_VIDEO_DIVISOR))
                    output_cv_frame_to_write = self.annotator.draw_annotations_on_image(
                        frame_bgr, last_known_annotations_for_drawing, font_scale_override=dynamic_font_scale_video
                    )
                writer.write(output_cv_frame_to_write) # Zaten BGR formatında
                if frame_idx % (int(L_FPS) * 5) == 0:
                    logger.info(f"  {frame_idx}/{total_f if total_f > 0 else '?'} frames processed...")
        finally:
            cap.release(); writer.release()
        logger.info(f"Video processing completed: {output_media_path}")
        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(all_frames_json_results, f, indent=4, ensure_ascii=False)
            logger.info(f"Video JSON results saved: {output_json_path}")
        except Exception as e:
            summary_texts_for_video.append(f"WARNING: Failed to save video JSON: {e}")
            logger.error(f"Failed to save video JSON to '{output_json_path}': {e}", exc_info=True)
            output_json_path = None
        final_summary = "\n---\n".join(summary_texts_for_video) if summary_texts_for_video else "No significant recognitions."
        return output_media_path, output_json_path, final_summary


    def close(self):
        """Closes the system and releases resources."""
        logger.info("Closing Object Recognition System...")
        if self.model_manager:
            self.model_manager.close()
        # FAISS index is managed by FaissDBManager, Python's GC will handle it unless specific close needed
        if self.config.DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("System resources released (attempted).")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    # Create a config instance (or use default_config imported from config.py)
    # For this example, assume default_config is suitable.
    # If you need custom config, instantiate: my_config = SystemConfig() and modify it.

    # Example: Override a config value if needed for testing
    # default_config.YOLO_CONFIDENCE_THRESHOLD = 0.3 
    
    # Ensure the default dataset path for enrollment exists if you run scan_and_enroll
    os.makedirs(default_config.DEFAULT_DATASET_BASE_DIR, exist_ok=True)
    # You might want to populate default_config.DEFAULT_DATASET_BASE_DIR with some test images
    # e.g., data/datasets/enrollment_data/car_brand_A/image1.jpg
    #       data/datasets/enrollment_data/car_brand_A/image2.jpg
    #       data/datasets/enrollment_data/car_brand_B/image1.jpg
    
    logger.info("Starting application __main__ block.")

    try:
        with ObjectRecognitionSystem(config=default_config) as system:
            # --- 1. (Optional) Scan filesystem to enroll objects ---
            # print("\n--- Scanning Filesystem for Enrollment ---")
            # scan_status, df_after_scan = system.scan_and_enroll_from_filesystem()
            # print(scan_status)
            # print("Enrolled objects after scan:")
            # print(df_after_scan.to_string())

            # --- 2. (Optional) Manually enroll an object (e.g. if no filesystem data) ---
            # print("\n--- Manual Enrollment Example ---")
            # try:
            #     # Create some dummy PIL images for testing manual enrollment
            #     # In a real scenario, these would be loaded from files or other sources
            #     dummy_image_data_1 = np.zeros((100, 100, 3), dtype=np.uint8)
            #     dummy_image_data_1[:, :, 0] = 255 # Red channel
            #     dummy_pil_1 = Image.fromarray(dummy_image_data_1)
            #     dummy_image_data_2 = np.zeros((120, 80, 3), dtype=np.uint8)
            #     dummy_image_data_2[:, :, 1] = 255 # Green channel
            #     dummy_pil_2 = Image.fromarray(dummy_image_data_2)
            #
            #     enroll_status, df_after_manual = system.enroll_object_manually(
            #         "TestObjectManual", [dummy_pil_1, dummy_pil_2]
            #     )
            #     print(enroll_status)
            #     print("Enrolled objects after manual add:")
            #     print(df_after_manual.to_string())
            # except Exception as e_manual:
            #     logger.error(f"Error during manual enrollment example: {e_manual}", exc_info=True)


            # --- 3. List enrolled objects ---
            print("\n--- Current Enrolled Objects ---")
            enrolled_df = system.get_enrolled_objects_dataframe()
            if not enrolled_df.empty:
                print(enrolled_df.to_string())
            else:
                print("No objects currently enrolled.")

            # # --- 4. Recognize objects in a single image ---
            # print("\n--- Image Recognition Example ---")
            # # Create a dummy test image or provide a path to a real image
            # test_image_path = "/home/earsal@ETE.local/Downloads/bwm2.jpeg"
            # # Create a dummy image if it doesn't exist for the example to run
            # if not os.path.exists(test_image_path):
            #     try:
            #         dummy_img_cv = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            #         cv2.imwrite(test_image_path, dummy_img_cv)
            #         logger.info(f"Created dummy OpenCV test image at: {test_image_path}")
            #     except Exception as e: logger.error(f"Could not create dummy CV test image: {e}")



            # if os.path.exists(test_image_path):
            #     current_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            #     processed_img_path, json_img_path, summary_img = system.recognize_image_and_save(
            #         input_image_path=test_image_path,
            #         processing_date_str=current_time_str,
            #         distance_threshold=0.2
            #     )
            #     print(f"Image Processing Summary:\n{summary_img}")
            #     if processed_img_path: print(f"Annotated Image: {processed_img_path}")
            #     if json_img_path: print(f"JSON Results: {json_img_path}")
            # else:
            #     print(f"Test image '{test_image_path}' not found. Skipping image recognition example.")

            # # --- 5. (Optional) Delete an enrolled object ---
            # # print("\n--- Deleting an Object Example ---")
            # # delete_status, was_deleted, df_after_delete = system.delete_enrolled_object("TestObjectManual") # Use a name that was enrolled
            # # print(delete_status)
            # # if was_deleted:
            # #     print("Enrolled objects after deletion:")
            # #     print(df_after_delete.to_string())


            # --- 6. (Optional) Recognize objects in a video ---
            print("\n--- Video Recognition Example ---")
            test_video_path = "/home/earsal@ETE.local/Downloads/C63 AMG Vs Bmw M3!!! - Motor Reviews (720p, h264).mp4"
            # Create a dummy video if it doesn't exist for the example to run
            if not os.path.exists(test_video_path):
                try:
                    frame_width, frame_height = 640, 480
                    out_cv = cv2.VideoWriter(test_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width, frame_height))
                    for _ in range(50): 
                        dummy_frame = np.random.randint(0, 256, (frame_height, frame_width, 3), dtype=np.uint8)
                        out_cv.write(dummy_frame)
                    out_cv.release()
                    logger.info(f"Created dummy test video at: {test_video_path}")
                except Exception as e_create_vid:
                    logger.error(f"Could not create dummy video: {e_create_vid}")

            if os.path.exists(test_video_path):
                current_time_str_vid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                processed_vid_path, json_vid_path, summary_vid = system.recognize_video_and_save(
                    video_path_input=test_video_path,
                    processing_date_str=current_time_str_vid,
                    distance_threshold=0.38,
                    process_every_n_frames=default_config.PROCESS_EVERY_N_FRAMES
                )
                print(f"Video Processing Summary:\n{summary_vid}")
                if processed_vid_path: print(f"Annotated Video: {processed_vid_path}")
                if json_vid_path: print(f"Video JSON Results: {json_vid_path}")
            else:
                print(f"Test video '{test_video_path}' not found. Skipping video recognition example.")

    except Exception as e:
        logger.critical(f"An unhandled error occurred in the main application: {e}", exc_info=True)
    finally:
        logger.info("Application __main__ block finished.")