"""Simple ADAS integration wrapper."""

from ..adas.system import ADASystem


def adas(model_path: str) -> str:
    system = ADASystem(model_path)
    return system.optimize_agent_architecture(model_path)
