import guidance
import pytest

def test_basic():
    """ Test the basic behavior of the LLaMA model.
    """

    # skip if no GPU
    import torch
    if torch.cuda.device_count() == 0:
        pytest.skip("No GPU, so skipping large model test.")

    # just make sure it runs
    llm = guidance.llms.transformers.LLaMA('elinas/llama-13b-hf-transformers-4.29', device=1)
    out = guidance("""The height of the Sears tower is {{gen 'answer' max_tokens=10}}""", llm=llm)()
    out = guidance(
        """The Sun is very {{#select 'answer'}}hot{{or}}cold{{/select}}.""", llm=llm
    )()
    assert len(out["answer"]) > 0