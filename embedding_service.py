"""
Purpose: This service, using the ObjectEmbedder class, takes an image (specifically,
 a crop of a detected object) and uses the ResNet model (obtained from ModelManager) to convert that image into a numerical feature vector (an embedding).
Key Idea: This vector is a compact representation that captures the important visual features of the object,
 allowing us to compare it to other objects. It also handles L2 normalization of these embeddings, which is crucial for calculating meaningful cosine similarity (or distance) in FAISS.
"""
import torch
import numpy as np
import faiss 
import logging
from PIL import Image
import cv2
from model_manager import ModelManager
from config import SystemConfig

logger = logging.getLogger(__name__)

class ObjectEmbedder:
    """
    Handles the generation of embeddings for object images using a pre-trained model.
    """
    def __init__(self, model_manager: ModelManager, config: SystemConfig):
        self.model_manager = model_manager
        self.config = config
        self.embedding_model, self.preprocess, self.device, self.ready = \
            self.model_manager.get_embedding_model_details()

    def get_object_embedding(self, object_cv_image: np.ndarray) -> np.ndarray | None:
        """
        Generates a normalized L2 embedding for a given OpenCV image (NumPy BGR array).

        Args:
            object_cv_image (np.ndarray): The input OpenCV image (BGR format).

        Returns:
            np.ndarray | None: A 1D numpy array representing the L2 normalized embedding,
                               or None if embedding fails.
        """
        if not self.ready or self.embedding_model is None or self.preprocess is None or self.device is None:
            logger.warning("Embedding model is not ready. Cannot generate embedding.")
            return None
        if object_cv_image is None:
            logger.warning("Input OpenCV image is None. Cannot generate embedding.")
            return None

        height, width = object_cv_image.shape[:2]
        if width < self.config.MIN_IMAGE_DIM_FOR_EMBEDDING or \
           height < self.config.MIN_IMAGE_DIM_FOR_EMBEDDING:
            logger.warning(f"Image size {width}x{height} is too small for embedding. "
                           f"Minimum is {self.config.MIN_IMAGE_DIM_FOR_EMBEDDING}x{self.config.MIN_IMAGE_DIM_FOR_EMBEDDING}.")
            return None
        try:
            # 1. OpenCV BGR'den RGB'ye dönüştür
            rgb_cv_image = cv2.cvtColor(object_cv_image, cv2.COLOR_BGR2RGB)

            # 2. TorchVision transformları PIL imajı beklediği için RGB NumPy dizisini PIL'e dönüştür
            pil_image = Image.fromarray(rgb_cv_image)

            # 3. Standart TorchVision preprocess adımlarını uygula
            img_t = self.preprocess(pil_image) # Preprocess PIL imajını alır
            batch_t = torch.unsqueeze(img_t, 0).to(self.device)

            with torch.no_grad():
                embedding = self.embedding_model(batch_t)

            embedding_np = embedding.squeeze().cpu().numpy().astype('float32')

            if np.isnan(embedding_np).any():
                logger.warning("Generated embedding contains NaN values.")
                return None

            embedding_np_2d = embedding_np.reshape(1, -1)
            faiss.normalize_L2(embedding_np_2d)
            return embedding_np_2d.flatten()

        except Exception as e:
            logger.error(f"Error generating object embedding: {e}, Image shape: {object_cv_image.shape}", exc_info=True)
            return None
