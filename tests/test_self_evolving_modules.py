import importlib.util
from pathlib import Path
import sys
import types

import pytest


def _load_self_evolving_system():
    """Import SelfEvolvingSystem with lightweight stubs."""
    if "agents" not in sys.modules:
        agents = types.ModuleType("agents")
        sys.modules["agents"] = agents

        # language model config stub
        lm_pkg = types.ModuleType("agents.language_models")
        openai_gpt = types.ModuleType("agents.language_models.openai_gpt")

        class OpenAIGPTConfig:
            def __init__(self, model_name: str = "gpt"):
                self.model_name = model_name

            def create(self):
                return object()

        openai_gpt.OpenAIGPTConfig = OpenAIGPTConfig
        sys.modules["agents.language_models"] = lm_pkg
        sys.modules["agents.language_models.openai_gpt"] = openai_gpt

        # quality assurance stub
        se_pkg = types.ModuleType("agents.self_evolve")
        qa_mod = types.ModuleType("agents.self_evolve.quality_assurance")

        class BasicUPOChecker:
            def __init__(self):
                self.upo_threshold = 0.5

        qa_mod.BasicUPOChecker = BasicUPOChecker
        sys.modules["agents.self_evolve"] = se_pkg
        sys.modules["agents.self_evolve.quality_assurance"] = qa_mod

        # utils stubs
        utils_mod = types.ModuleType("agents.utils")

        class DirectPreferenceOptimizer:
            pass

        class DPOConfig:
            pass

        class MCTSConfig:
            def __init__(self):
                self.exploration_weight = 1.0
                self.simulation_depth = 1

        class MonteCarloTreeSearch:
            pass

        utils_mod.DirectPreferenceOptimizer = DirectPreferenceOptimizer
        utils_mod.DPOConfig = DPOConfig
        utils_mod.MCTSConfig = MCTSConfig
        utils_mod.MonteCarloTreeSearch = MonteCarloTreeSearch
        sys.modules["agents.utils"] = utils_mod

        # task stub
        task_mod = types.ModuleType("agents.utils.task")

        class Task:
            def __init__(self, content: str, task_type: str = "task"):
                self.content = content
                self.type = task_type

        task_mod.Task = Task
        sys.modules["agents.utils.task"] = task_mod

        # core communication stubs
        comm_mod = types.ModuleType("core.communication")

        class Message:
            pass

        class MessageType:
            pass

        class Priority:
            pass

        class StandardCommunicationProtocol:
            def subscribe(self, *_args, **_kwargs):
                pass

            async def send_message(self, *_args, **_kwargs):
                pass

        comm_mod.Message = Message
        comm_mod.MessageType = MessageType
        comm_mod.Priority = Priority
        comm_mod.StandardCommunicationProtocol = StandardCommunicationProtocol
        sys.modules["core.communication"] = comm_mod

        # core error handling stubs
        err_mod = types.ModuleType("core.error_handling")

        class AIVillageError(Exception):
            pass

        class ErrorCategory:
            INITIALIZATION = "init"
            VALIDATION = "val"
            PROCESSING = "proc"
            EVOLUTION = "evo"

        class ErrorSeverity:
            CRITICAL = "crit"
            WARNING = "warn"
            ERROR = "err"

        def get_component_logger(_name, _extra=None):
            class Logger:
                def info(self, *args, **kwargs):
                    pass

                def error(self, *args, **kwargs):
                    pass

                def debug(self, *args, **kwargs):
                    pass

            return Logger()

        def with_error_handling(*_args, **_kwargs):
            def decorator(func):
                return func

            return decorator

        err_mod.AIVillageException = AIVillageError
        err_mod.ErrorCategory = ErrorCategory
        err_mod.ErrorSeverity = ErrorSeverity
        err_mod.get_component_logger = get_component_logger
        err_mod.with_error_handling = with_error_handling
        sys.modules["core.error_handling"] = err_mod

        # rag system stubs
        core_config = types.ModuleType("rag_system.core.config")

        class UnifiedConfig:
            pass

        core_config.UnifiedConfig = UnifiedConfig
        sys.modules["rag_system.core.config"] = core_config

        pipeline_mod = types.ModuleType("rag_system.core.pipeline")

        class EnhancedRAGPipeline:
            def __init__(self, *args, **kwargs):
                pass

        pipeline_mod.EnhancedRAGPipeline = EnhancedRAGPipeline
        sys.modules["rag_system.core.pipeline"] = pipeline_mod

        vector_mod = types.ModuleType("rag_system.retrieval.vector_store")

        class VectorStore:
            pass

        vector_mod.VectorStore = VectorStore
        sys.modules["rag_system.retrieval.vector_store"] = vector_mod

        tracker_mod = types.ModuleType("rag_system.tracking.unified_knowledge_tracker")

        class UnifiedKnowledgeTracker:
            pass

        tracker_mod.UnifiedKnowledgeTracker = UnifiedKnowledgeTracker
        sys.modules["rag_system.tracking.unified_knowledge_tracker"] = tracker_mod
        utils_pkg = types.ModuleType("rag_system.utils")
        logging_mod = types.ModuleType("rag_system.utils.logging")

        def setup_logger(_name):
            class Logger:
                def info(self, *args, **kwargs):
                    pass

            return Logger()

        logging_mod.setup_logger = setup_logger
        sys.modules["rag_system.utils"] = utils_pkg
        sys.modules["rag_system.utils.logging"] = logging_mod

    spec = importlib.util.spec_from_file_location(
        "unified_base_agent",
        Path(__file__).resolve().parents[1] / "experimental" / "agents" / "agents" / "unified_base_agent.py",
    )
    uba = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(uba)
    return uba.SelfEvolvingSystem


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None,
    reason="torch and transformers are required",
)
def test_quiet_star_integration():
    ses_cls = _load_self_evolving_system()
    ses = ses_cls([])

    qs_spec = importlib.util.spec_from_file_location(
        "quiet_star",
        Path(__file__).resolve().parents[1] / "experimental" / "training" / "training" / "quiet_star.py",
    )
    qs_module = importlib.util.module_from_spec(qs_spec)
    assert qs_spec.loader is not None
    qs_spec.loader.exec_module(qs_module)
    quiet_cls = qs_module.QuietSTaRModel

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    ses.quiet_star = quiet_cls(base_model)

    tokens = tokenizer.encode("Hello", return_tensors="pt")
    try:
        logits, thought_logits = ses.quiet_star(tokens, generate_thoughts=False)
    except Exception as e:  # noqa: BLE001 - diagnostic for broken module
        pytest.fail(f"Quiet-STaR execution failed: {e}")
    assert logits.shape[0] == 1
    assert thought_logits is None


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is required",
)
def test_expert_vectors_integration():
    ses_cls = _load_self_evolving_system()
    ses = ses_cls([])

    ev_spec = importlib.util.spec_from_file_location(
        "expert_vectors",
        Path(__file__).resolve().parents[1] / "experimental" / "training" / "training" / "expert_vectors.py",
    )
    ev_module = importlib.util.module_from_spec(ev_spec)
    assert ev_spec.loader is not None
    ev_spec.loader.exec_module(ev_module)
    ev_cls = ev_module.ExpertVectorSystem
    import torch

    model = torch.nn.Linear(4, 4, bias=False)
    ses.expert_vectors = ev_cls(model)
    vector = ses.expert_vectors.train_expert_vector_svf("v", scale=0.01)
    before = model.weight.clone()
    ses.expert_vectors.apply_expert_vector(vector, scaling=1.0)
    after = model.weight
    assert not torch.allclose(before, after), "Expert vector application had no effect"


def test_adas_integration(tmp_path):
    ses_cls = _load_self_evolving_system()
    ses = ses_cls([])

    adas_spec = importlib.util.spec_from_file_location(
        "adas_system",
        Path(__file__).resolve().parents[1] / "src" / "agent_forge" / "adas" / "system.py",
    )
    adas_module = importlib.util.module_from_spec(adas_spec)
    assert adas_spec.loader is not None
    adas_spec.loader.exec_module(adas_module)
    adas_cls = adas_module.ADASystem

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "weights.bin").write_text("data")
    ses.adas_optimizer = adas_cls(str(model_dir))
    out_dir = tmp_path / "out"
    optimized_path = ses.adas_optimizer.optimize_agent_architecture(str(out_dir), iterations=2)
    assert (Path(optimized_path) / "weights.bin").exists(), "ADAS optimization did not produce expected output"
