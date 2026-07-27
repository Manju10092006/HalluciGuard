import logging
import threading
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseModelWrapper(ABC):
    """
    Abstract base class for all model wrappers.
    Provides thread-safe lazy loading and fallback handling.
    """
    
    def __init__(self, model_id: str):
        self._model_id = model_id
        self._is_loaded = False
        self._is_available = True
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        
    @property
    def model_id(self) -> str:
        return self._model_id
        
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
        
    @property
    def device(self) -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
            
    def load(self) -> None:
        """Thread-safe lazy loading of the model."""
        if self._is_loaded or not self._is_available:
            return
            
        with self._lock:
            if self._is_loaded or not self._is_available:
                return
                
            try:
                self._load_internal()
                self._is_loaded = True
                logger.info(f"Successfully loaded model {self._model_id}")
            except Exception as e:
                self._is_available = False
                logger.error(f"Failed to load model {self._model_id}: {e}")
                
    @abstractmethod
    def _load_internal(self) -> None:
        """Actual implementation of loading the model."""
        pass
        
    def unload(self) -> None:
        """Clear model from memory."""
        with self._lock:
            self._model = None
            self._tokenizer = None
            self._pipeline = None
            self._is_loaded = False
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info(f"Unloaded model {self._model_id}")

    def ensure_loaded(self):
        if not self._is_loaded and self._is_available:
            self.load()
