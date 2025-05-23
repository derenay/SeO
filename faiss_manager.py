"""
Purpose: The FaissDBManager class is the heart of the object memory. It manages the FAISS index,
 which is a specialized database for efficiently searching through these numerical feature vectors.
Key Idea: It handles:

    Loading an existing FAISS index and object metadata from disk.
    Saving the current index and metadata.
    Adding new object prototypes (their average embeddings) to the index.
    Deleting object prototypes.
    Performing similarity searches: given an embedding of a detected object, it finds the most similar known objects in its database.
    It also includes a threading.Lock to ensure that operations that modify the index (like adding or deleting objects) are thread-safe, which is important if you were to use this system in a multi-threaded application.
"""

import faiss
import numpy as np
import json
import os
import threading
import logging
from typing import List, Dict, Tuple, Any, Optional 

from config import SystemConfig

logger = logging.getLogger(__name__)

class FaissDBManager:
    """
    Manages the FAISS index for object prototypes, including loading, saving,
    adding, deleting, and searching.
    """
    def __init__(self, config: SystemConfig):
        self.config = config
        self.index: faiss.Index | None = None
        self.id_to_metadata: List[Dict[str, Any]] = []
        self.object_name_to_faiss_id: Dict[str, int] = {}
        self._lock = threading.Lock()  

        self._load_data()

    def _initialize_structures(self):
        """Initializes empty FAISS index and metadata."""
        logger.info("Initializing new empty FAISS index and metadata structures.")
        self.index = faiss.IndexFlatIP(self.config.EMBEDDING_DIMENSION)
        self.id_to_metadata = []
        self.object_name_to_faiss_id = {}

    def _load_data(self):
        """Loads FAISS index and metadata from disk."""
        with self._lock:
            loaded_idx_success = False
            if os.path.exists(self.config.FAISS_INDEX_FILE_PATH):
                try:
                    self.index = faiss.read_index(self.config.FAISS_INDEX_FILE_PATH)
                    logger.info(f"FAISS index '{self.config.FAISS_INDEX_FILE_PATH}' loaded "
                                f"({self.index.ntotal} vectors).")
                    loaded_idx_success = True
                except Exception as e:
                    logger.error(f"Failed to load FAISS index: {e}. Initializing new index.", exc_info=True)
                    self.index = None 

            if os.path.exists(self.config.ID_MAPPING_FILE_PATH):
                try:
                    with open(self.config.ID_MAPPING_FILE_PATH, 'r', encoding='utf-8') as f:
                        map_data = json.load(f)
                    self.id_to_metadata = map_data.get('id_to_metadata', [])
                    self.object_name_to_faiss_id = map_data.get('name_to_id', {})
                    logger.info(f"ID map '{self.config.ID_MAPPING_FILE_PATH}' loaded "
                                f"({len(self.id_to_metadata)} records).")

                    if loaded_idx_success and self.index is not None and \
                       self.index.ntotal != len(self.id_to_metadata):
                        logger.warning("FAISS index and ID map sizes are inconsistent! Resetting structures.")
                        self._initialize_structures() # Reset both
                        return 
                    if not loaded_idx_success and self.id_to_metadata: # Map exists but index failed/missing
                        logger.warning("ID map exists but FAISS index is missing or failed to load! Resetting structures.")
                        self._initialize_structures() # Reset both
                        return 

                except Exception as e:
                    logger.error(f"Failed to load ID map: {e}. This may affect consistency if index was loaded.", exc_info=True)
                    # If index loaded but map failed, it's an inconsistent state.
                    if loaded_idx_success:
                        logger.warning("Index loaded but map failed. Resetting for safety.")
                        self._initialize_structures()
                        return

  
            # never existed, or failed to load and no map to trigger reset
            if self.index is None:
                self._initialize_structures()

    def _save_data(self) -> bool:
        """Saves FAISS index and metadata to disk. Assumes lock is already held."""
        if self.index is None:
            logger.error("No FAISS index to save.")
            return False
        try:
            faiss.write_index(self.index, self.config.FAISS_INDEX_FILE_PATH)
            logger.info(f"FAISS index saved to '{self.config.FAISS_INDEX_FILE_PATH}' ({self.index.ntotal} vectors).")

            map_data = {
                'id_to_metadata': self.id_to_metadata,
                'name_to_id': self.object_name_to_faiss_id
            }
            with open(self.config.ID_MAPPING_FILE_PATH, 'w', encoding='utf-8') as f:
                json.dump(map_data, f, indent=4, ensure_ascii=False)
            logger.info(f"ID map saved to '{self.config.ID_MAPPING_FILE_PATH}'.")
            return True
        except Exception as e:
            logger.error(f"Error saving FAISS data: {e}", exc_info=True)
            return False

    def _rebuild_index_and_metadata(self, prototypes_map: Dict[str, Dict[str, Any]]) -> bool:
        """
        Helper function to rebuild the FAISS index and metadata from a dictionary of prototypes.
        Assumes lock is held.

        Args:
            prototypes_map: Dict where keys are object names and values are dicts
                            with 'prototype_vector', 'num_samples', 'source'.
        """
        new_vectors_list = []
        new_id_to_meta = []
        new_name_to_id = {}
        current_faiss_id = 0

        for name, data in prototypes_map.items():
            new_vectors_list.append(data['prototype_vector']) 
            new_id_to_meta.append({
                'name': name,
                'num_samples': data['num_samples'],
                'source': data['source']
            })
            new_name_to_id[name] = current_faiss_id
            current_faiss_id += 1

        try:
            if not new_vectors_list:
                logger.info("No prototypes to build index from. Initializing empty structures.")
                self._initialize_structures() 
                return True 

            all_vectors_np = np.array(new_vectors_list).astype('float32')
            if all_vectors_np.ndim == 1 and len(new_vectors_list) == 1: # Single vector
                all_vectors_np = all_vectors_np.reshape(1, -1)
            elif all_vectors_np.ndim != 2 or all_vectors_np.shape[1] != self.config.EMBEDDING_DIMENSION:
                logger.error(f"Vector array shape mismatch: {all_vectors_np.shape}. Expected N x {self.config.EMBEDDING_DIMENSION}")
                # Don't modify existing index if new data is bad
                return False

            # Vectors should already be L2 normalized before being passed here.
            # Re-normalizing can be a safeguard if unsure.
            # faiss.normalize_L2(all_vectors_np)

            rebuilt_faiss_index = faiss.IndexFlatIP(self.config.EMBEDDING_DIMENSION)
            rebuilt_faiss_index.add(all_vectors_np)

            self.index = rebuilt_faiss_index
            self.id_to_metadata = new_id_to_meta
            self.object_name_to_faiss_id = new_name_to_id
            logger.info(f"FAISS index rebuilt with {self.index.ntotal} vectors.")
            return True
        except Exception as e:
            logger.error(f"Error rebuilding FAISS index from prototypes: {e}", exc_info=True)
            return False 

    def add_or_update_prototype(self, object_name: str, prototype_vector: np.ndarray,
                                num_samples: int, source: str) -> bool:
        """Adds a new prototype or updates an existing one, then rebuilds and saves."""
        with self._lock:
            current_prototypes = {}
            if self.index and self.index.ntotal > 0:
                for fid, meta in enumerate(self.id_to_metadata):
                    if fid < self.index.ntotal: 
                        vec = self.index.reconstruct(fid)
                        current_prototypes[meta['name']] = {
                            'prototype_vector': vec,
                            'num_samples': meta['num_samples'],
                            'source': meta['source']
                        }
                    else: 
                        logger.warning(f"Metadata ID {fid} out of bounds for FAISS index (total {self.index.ntotal}). Skipping.")


            current_prototypes[object_name.strip()] = {
                'prototype_vector': prototype_vector.flatten(), # Ensure 1D
                'num_samples': num_samples,
                'source': source
            }

            if not self._rebuild_index_and_metadata(current_prototypes):
                logger.error(f"Failed to rebuild index after adding/updating '{object_name}'. Original index may still be active.")
                return False
            return self._save_data()

    def delete_prototype(self, object_name_to_delete: str) -> bool:
        """Deletes a prototype, then rebuilds and saves."""
        with self._lock:
            if not object_name_to_delete or object_name_to_delete not in self.object_name_to_faiss_id:
                logger.warning(f"Object '{object_name_to_delete}' not found for deletion.")
                return False

            logger.info(f"Deleting '{object_name_to_delete}'. FAISS index will be rebuilt.")
            
            remaining_prototypes = {}
            if self.index and self.index.ntotal > 0:
                for fid, meta in enumerate(self.id_to_metadata):
                    if meta['name'] == object_name_to_delete:
                        continue
                    if fid < self.index.ntotal:
                        vec = self.index.reconstruct(fid)
                        remaining_prototypes[meta['name']] = {
                            'prototype_vector': vec,
                            'num_samples': meta['num_samples'],
                            'source': meta['source']
                        }
            
            if not self._rebuild_index_and_metadata(remaining_prototypes):
                logger.error(f"Failed to rebuild index after deleting '{object_name_to_delete}'. Original index may still be active.")
                return False
            return self._save_data()

    def replace_all_prototypes(self, new_prototypes_map: Dict[str, Dict[str, Any]]) -> bool:
        """Replaces all existing prototypes with a new set. Used by filesystem scan."""
        with self._lock:
            logger.info(f"Replacing all prototypes with {len(new_prototypes_map)} new entries.")
            if not self._rebuild_index_and_metadata(new_prototypes_map):
                logger.error("Failed to rebuild index for replacing all prototypes.")
                return False
            return self._save_data()

    def search(self, query_embedding: np.ndarray, k: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Searches the FAISS index for the k nearest neighbors.
        The lock should be handled by the caller if a consistent view of index + metadata is needed.
        This method itself is thread-safe for read if index object is not being replaced.
        For simplicity in this refactor, we'll make the caller get a "snapshot" of data.
        """
        # This method does not use the internal lock to allow for concurrent reads
        # if the caller manages the consistency of the index reference.
        # The ObjectRecognitionSystem will get current_index and current_metadata under lock.
        current_index = self.index # Get current reference (could change)

        if current_index is None or current_index.ntotal == 0:
            return None, None
        if query_embedding is None:
            return None, None
        
        try:
            # Query embedding should be L2 normalized and 2D (1, D)
            query_embedding_2d = query_embedding.reshape(1, -1).astype('float32')
            # Assuming query_embedding is already normalized by ObjectEmbedder
            distances, indices = current_index.search(query_embedding_2d, k)
            return distances, indices
        except Exception as e:
            logger.error(f"FAISS search error: {e}", exc_info=True)
            return None, None
            
    def get_current_search_state(self) -> Tuple[Optional[faiss.Index], List[Dict[str, Any]]]:
        """
        Returns a snapshot of the current FAISS index reference and metadata list
        under lock, for safe concurrent searching.
        """
        with self._lock:
            # Return a reference to the index and a shallow copy of the metadata list
            return self.index, list(self.id_to_metadata)


    def get_all_metadata_summary(self) -> List[Dict[str, Any]]:
        """Returns a summary of all enrolled objects."""
        with self._lock:
            summary = []
            for idx, meta_data in enumerate(self.id_to_metadata):
                summary.append({
                    "name": meta_data['name'],
                    "num_samples": meta_data['num_samples'],
                    "source": meta_data.get('source', 'Unknown'),
                    "faiss_id": idx # This is the current internal ID based on list order
                })
            return summary

    def get_index_ntotal(self) -> int:
        with self._lock:
            return self.index.ntotal if self.index else 0