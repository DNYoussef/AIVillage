# Fog gateway scheduler package initialization  
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'infrastructure'))