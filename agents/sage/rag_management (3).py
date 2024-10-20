from typing import Dict, Any
from ai_village.utils.exceptions import AIVillageException
from ai_village.utils.prompt_template import PromptTemplate
import logging
import asyncio
from ai_village.core.sage import Sage
from ai_village.utils.ai_provider import AIProvider

logger = logging.getLogger(__name__)

class KingRAGManagement:
    def __init__(self):
        self.sage = Sage()
        self.ai_provider = AIProvider()
        self.prompt_template = PromptTemplate()

    # ... (rest of the methods from the original rag_management.py)