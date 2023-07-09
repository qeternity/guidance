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

    model_path = "elinas/llama-13b-hf-transformers-4.29"
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    model.tie_weights()
    model = load_checkpoint_and_dispatch(
    model, model_path, device_map="auto", dtype=torch.float16
)

    # just make sure it runs
    llm = guidance.llms.transformers.LLaMA(model, tokenizer, device=1, torch_dtype=torch.float16)
    out = guidance("""The height of the Sears tower is {{gen 'answer' max_tokens=10}}""", llm=llm)()
    out = guidance(
        """The Sun is very {{#select 'answer'}}hot{{or}}cold{{/select}}.""", llm=llm
    )()
    assert len(out["answer"]) > 0