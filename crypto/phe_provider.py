import phe
import json
from base import BaseCryptoProvider

class PHEProvider(BaseCryptoProvider):
    def __init__(self, public_key, private_key=None):
        self.public_key = public_key
        self.private_key = private_key

    @classmethod
    def generate(cls):
        pub, priv = phe.paillier.generate_paillier_keypair()
        return cls(pub, priv)

    def encrypt(self, value: float):
        return self.public_key.encrypt(value)

    def decrypt(self, encrypted):
        if not self.private_key:
            raise ValueError("No private key provided")
        return self.private_key.decrypt(encrypted)

    def save_keys(self, path: str):
        with open(path, 'w') as f:
            json.dump({
                'n': str(self.public_key.n),
                'private': self.private_key.to_json() if self.private_key else None
            }, f)

    @classmethod
    def load_keys(cls, path: str):
        with open(path) as f:
            keys = json.load(f)
        pub = phe.paillier.PaillierPublicKey(n=int(keys['n']))
        priv = phe.paillier.PaillierPrivateKey(pub, **json.loads(keys['private'])) if keys['private'] else None
        return cls(pub, priv)
