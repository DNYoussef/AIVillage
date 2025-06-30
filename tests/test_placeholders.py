import pytest
import importlib

missing = []
for pkg in ["openai", "tiktoken", "transformers", "torch", "grokfast"]:
    if importlib.util.find_spec(pkg) is None:
        missing.append(pkg)

if missing:
    pytest.skip(f"Dependencies missing: {', '.join(missing)}", allow_module_level=True)

from agents.unified_base_agent import SelfEvolvingSystem


@pytest.mark.xfail(reason="Quiet-STaR not implemented")
@pytest.mark.asyncio
async def test_quiet_star_module():
    ses = SelfEvolvingSystem([])
    assert hasattr(ses, "quiet_star"), "Quiet-STaR module missing"


@pytest.mark.xfail(reason="Expert vectors not implemented")
def test_expert_vectors_module():
    ses = SelfEvolvingSystem([])
    assert hasattr(ses, "expert_vectors"), "Expert vectors module missing"


@pytest.mark.xfail(reason="ADAS optimization not implemented")
def test_adas_module():
    ses = SelfEvolvingSystem([])
    assert hasattr(ses, "adas_optimizer"), "ADAS module missing"
