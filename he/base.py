from abc import ABC, abstractmethod
from typing import List, Any, Dict

class BaseCryptoProvider(ABC):
    @abstractmethod
    def encrypt(self, value: float) -> Dict[str, Any]:
        pass

    @abstractmethod
    def decrypt(self, encrypted_dict: Dict[str, Any]) -> float:
        pass
    
    def encrypt_weights(self, weights: List[float]) -> List[Any]:
        return [self.encrypt(w) for w in weights]

    def decrypt_weights(self, weights: List[Any]) -> List[float]:
        return [self.decrypt(w) for w in weights]
