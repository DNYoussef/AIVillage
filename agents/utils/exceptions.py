class AIVillageException(Exception):
    """Custom exception class for AI Village-specific errors."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
