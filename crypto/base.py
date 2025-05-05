from abc import ABC, abstractmethod
from typing import List, Any

class BaseCryptoProvider(ABC):
    @abstractmethod
    def encrypt(self, value: float) -> Any: ...
    @abstractmethod
    def decrypt(self, value: Any) -> float: ...
    
    def encrypt_weights(self, weights: List[float]) -> List[Any]:
        return [self.encrypt(w) for w in weights]

    def decrypt_weights(self, weights: List[Any]) -> List[float]:
        return [self.decrypt(w) for w in weights]
