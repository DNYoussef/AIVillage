from .bitlinearization import BitlinearizationTask, BitNetModel, BitLinear
from .hyperparameter_compression import HyperCompressionTask, HyperCompressor
from .hyperparamter_compression import HyperparamterCompressionTask

__all__ = [
    'BitlinearizationTask',
    'BitNetModel',
    'BitLinear',
    'HyperCompressionTask',
    'HyperCompressor',
    'HyperparamterCompressionTask'
]

# Add a warning message about the typo in the file name
import warnings

warnings.warn("The 'hyperparamter_compression.py' file contains a typo in its name. Consider renaming it to 'hyperparameter_compression.py' or removing it if it's redundant.", DeprecationWarning)
