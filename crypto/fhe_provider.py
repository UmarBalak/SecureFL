try:
    import concrete.numpy as cnp
    import numpy as np
except ImportError:
    raise ImportError("Install concrete-numpy for FHE support.")

from base import BaseCryptoProvider

class FHEProvider(BaseCryptoProvider):
    def __init__(self):
        def identity_op(x):
            return x
        self.compiler = cnp.Compiler(identity_op)
        self.inputset = [np.array([float(i)]) for i in range(10)]
        self.circuit = self.compiler.compile(self.inputset)

    def encrypt(self, value: float):
        return self.circuit.encrypt(np.array([value]))

    def decrypt(self, encrypted):
        return float(self.circuit.decrypt(encrypted)[0])
