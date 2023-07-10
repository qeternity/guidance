import guidance
import pytest
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

def test_basic():
    """ Test the basic behavior of the LLaMA model.
    """

    # skip if no GPU
    import torch
    if torch.cuda.device_count() == 0:
        pytest.skip("No GPU, so skipping large model test.")

    model_path = "/root/.cache/huggingface/hub/models--elinas--llama-7b-hf-transformers-4.29/snapshots/d33594ee64ef1b6264543b6a88f60982a55fdb7a/"
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    model.tie_weights()
    model = load_checkpoint_and_dispatch(
        model, model_path, device_map="sequential", dtype=torch.float16
    )

    # just make sure it runs
    guidance.llm = guidance.llms.transformers.LLaMA(model, tokenizer, device='cuda', torch_dtype=torch.float16, caching=False)

    out = guidance("""The height of the Sears tower is {{gen 'answer' pattern='[0-9]+'}}""")()
    assert 1000 < int(out['answer']) < 10000

    # out = guidance("""The Sun is very {{#select 'answer'}}hot{{or}}cold{{/select}}.""")()
    # assert out["answer"] == 'hot'

    # out = guidance("""The North Pole is {{#select 'answer'}}scorching{{or}}freezing{{/select}}.""")()
    # assert out["answer"] == 'freezing'