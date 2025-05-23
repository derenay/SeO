"""
Purpose: This file is your central control panel for the entire system. It holds all the important settings and parameters,
 like paths to model files, threshold values, directory names for data and results, font settings, etc.
Key Idea: Instead of scattering settings throughout the code, they are all defined in the SystemConfig class here. 
This makes it easy to change a setting without hunting through multiple files.
"""
import os
import torch

class SystemConfig:
    """
    Centralized configuration for the Object Recognition System.
    """
    K_SHOTS_ENROLLMENT_MANUAL: int = 3
    MAX_IMAGES_PER_OBJECT_FS: int = 10 # Max images per object for filesystem enrollment
    MIN_IMAGE_DIM_FOR_EMBEDDING: int = 16 # Min width/height for an image to be embedded

    # Model Configurations
    EMBEDDING_MODEL_DESCRIPTION: str = 'ResNet50 (ImageNet)'
    YOLO_MODEL_NAME: str = 'yoloe-11l-seg-pf.pt' 
    EMBEDDING_DIMENSION: int = 2048  

    # FAISS and Recognition Settings
    DEFAULT_DISTANCE_THRESHOLD: float = 0.9
    FAISS_SEARCH_TOP_K: int = 1

    # Paths - Consider making these relative to a project root or fully configurable
    # PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__)) 
    PROJECT_ROOT: str = "project-02/yenidenemem_few_shot/object_teslim/save_2"

    # Adjusted paths to be more organized, e.g., within a 'data' subdirectory
    # DEFAULT_FAISS_INDEX_DIR: str = os.path.join(PROJECT_ROOT, "data", "faiss_indices")
    DEFAULT_FAISS_INDEX_DIR: str = "project-02/yenidenemem_few_shot/object_teslim"
    DEFAULT_FAISS_INDEX_FILENAME: str = "object_prototypes.faissidx"
    DEFAULT_ID_MAPPING_FILENAME: str = "object_id_map.json"

    # DEFAULT_DATASET_BASE_DIR: str = os.path.join(PROJECT_ROOT, "data", "datasets", "enrollment_data") 
    DEFAULT_DATASET_BASE_DIR: str = "project-02/yenidenemem_few_shot/object_teslim/save_2"
    OUTPUT_BASE_DIR: str = os.path.join(PROJECT_ROOT, "recognition_results")

    # Font paths
    FONT_PATH_WINDOWS: str = "arial.ttf"
    FONT_PATH_LINUX: str = "DejaVuSans.ttf"
    OPENCV_FONT_FACE: int = 0 
    OPENCV_FONT_SCALE_SMALL: float = 0.5
    OPENCV_FONT_SCALE_VIDEO_DIVISOR: int = 1200 # h / 1200 or w / 1200 for font scale
    OPENCV_TEXT_THICKNESS: int = 1
    ANNOTATION_BOX_THICKNESS_DIVISOR: int = 250 # h / 250 for line thickness
    # Technical Configurations
    TORCH_DEVICE_OVERRIDE: str | None = "cuda"  # e.g., "cuda", "cpu", None for auto
    YOLO_CONFIDENCE_THRESHOLD: float = 0.30
    # For video processing
    PROCESS_EVERY_N_FRAMES: int = 4 

    # --- Derived or Runtime-determined Configurations ---
    FAISS_INDEX_FILE_PATH: str
    ID_MAPPING_FILE_PATH: str
    FONT_PATH: str | None = None 

    def __init__(self):
        """
        Initializes configuration, creates necessary directories, and sets up environment.
        """
 
        os.makedirs(self.DEFAULT_FAISS_INDEX_DIR, exist_ok=True)
        os.makedirs(self.DEFAULT_DATASET_BASE_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_BASE_DIR, exist_ok=True)

        self.FAISS_INDEX_FILE_PATH = os.path.join(self.DEFAULT_FAISS_INDEX_DIR, self.DEFAULT_FAISS_INDEX_FILENAME)
        self.ID_MAPPING_FILE_PATH = os.path.join(self.DEFAULT_FAISS_INDEX_DIR, self.DEFAULT_ID_MAPPING_FILENAME)

        if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

       
        if self.TORCH_DEVICE_OVERRIDE:
            self.DEVICE = torch.device(self.TORCH_DEVICE_OVERRIDE)
        else:
            self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


default_config = SystemConfig()