"""
Purpose: Responsible for loading and managing the machine learning models – specifically,
 the YOLO model for object detection and the ResNet model for creating feature embeddings.

Key Idea: It ensures models are loaded correctly (e.g., onto the GPU if available)
 and provides a clean way for other parts of the system to access these models.

"""
import torch
import torchvision.models as models
from ultralytics import YOLOE
from typing import Tuple, Any
import logging

from config import SystemConfig

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages the loading and accessibility of machine learning models (YOLO, ResNet).
    """
    def __init__(self, config: SystemConfig):
        self.config = config
        self.yolo_model: YOLOE | None = None
        self.embedding_model_resnet: models.ResNet | None = None
        self.preprocess_resnet: Any = None # torchvision.transforms.Compose
        self.resnet_ready: bool = False

        self._initialize_yolo_model()
        self._initialize_embedding_model()

    def _initialize_yolo_model(self):
        """Initializes and loads the YOLO object detection model."""
        if self.yolo_model is not None:
            logger.info("YOLO model already initialized.")
            return
        try:
            self.yolo_model = YOLOE(self.config.YOLO_MODEL_NAME)
            self.yolo_model.to(self.config.DEVICE) 
            logger.info(f"YOLO model '{self.config.YOLO_MODEL_NAME}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model '{self.config.YOLO_MODEL_NAME}': {e}", exc_info=True)
            self.yolo_model = None

    def _initialize_embedding_model(self):
        """Initializes and loads the ResNet embedding model."""
        if self.resnet_ready:
            logger.info("ResNet embedding model already initialized.")
            return
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            self.embedding_model_resnet = models.resnet50(weights=weights)
            self.embedding_model_resnet.fc = torch.nn.Identity()  # Use as a feature extractor
            self.embedding_model_resnet.to(self.config.DEVICE)
            self.embedding_model_resnet.eval()
            self.preprocess_resnet = weights.transforms()
            self.resnet_ready = True
            logger.info(f"ResNet50 embedding model '{self.config.EMBEDDING_MODEL_DESCRIPTION}' "
                        f"loaded successfully on {self.config.DEVICE}.")
        except Exception as e:
            logger.error(f"Failed to load ResNet50 embedding model: {e}", exc_info=True)
            self.resnet_ready = False
            self.embedding_model_resnet = None
            self.preprocess_resnet = None

    def get_yolo_model(self) -> YOLOE | None:
        return self.yolo_model

    def get_embedding_model_details(self) -> Tuple[models.ResNet | None, Any, torch.device | None, bool]:
        """Returns the embedding model, preprocessing transforms, device, and ready status."""
        return self.embedding_model_resnet, self.preprocess_resnet, self.config.DEVICE, self.resnet_ready

    def close(self):
        """Releases model resources (if applicable)."""
        self.yolo_model = None
        self.embedding_model_resnet = None
        self.preprocess_resnet = None
        self.resnet_ready = False
        if self.config.DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        logger.info("ModelManager resources released.")