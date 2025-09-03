import pytest
import torch

try:  # pragma: no cover - runtime dependency check
    from core.agent_forge.integration.cogment.hf_export import CogmentHFConfig, CogmentForCausalLM
    from core.agent_forge.models.cogment.core.config import CogmentConfig
except ModuleNotFoundError:  # pragma: no cover - skip if Cogment not available
    CogmentHFConfig = CogmentForCausalLM = CogmentConfig = None


@pytest.mark.skipif(CogmentHFConfig is None, reason="Cogment dependencies not available")
def test_cogment_for_causallm_returns_states_and_attentions():
    cog_config = CogmentConfig(d_model=32, n_layers=2, n_head=2, vocab_size=128, max_refinement_steps=2)
    hf_config = CogmentHFConfig.from_cogment_config(cog_config)
    model = CogmentForCausalLM(hf_config)

    input_ids = torch.randint(0, cog_config.vocab_size, (1, 4))

    outputs = model(
        input_ids,
        output_hidden_states=True,
        output_attentions=True,
        max_refinement_steps=2,
    )

    assert outputs.hidden_states is not None
    assert outputs.attentions is not None
    assert len(outputs.hidden_states) == 2
    assert outputs.hidden_states[0].shape == (1, 4, cog_config.d_model)
    assert outputs.attentions[0].shape == (1, 4, 1)
