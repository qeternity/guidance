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
    guidance.llm = guidance.llms.transformers.LLaMA(model, tokenizer, torch_dtype=torch.float16, caching=False)

    out = guidance("""
The following are requests for fuel pricing for maritime vessels as conversations.
Following the conversation is the parsed data from the conversation for the fuel request.

### Start
a: Our good lady Lodesta Capella is completing discharging at Mersin and will go to dry dock at Tuzla.
Could you check the indicative prices for two options:
1) alongside at Mersin
2) Barge supply Istanbul-Tuzla range

50 mts MGO
b: No problem
b: CIA or need some credit terms?
a: COD or upto 7dd
b: ok
a: thanks
b: What at her dates in Mersin?
a: ETC 23-25 Apr
a: ETA Tuzla 29-30 Apr
b: tks
### End
```json
{
    "name": "Lady Lodesta Capella",
    "destination": "Mersin",
    "dates": "Apr 23-25",
    "quantity": "1000mt"
    "grade": "MGO",
    "terms": "COD or up to 7dd"
}
```

### Start
a: looking to get a quote to refuel the olympus
a: needs 1000 tons vlsfo at ara in a weeks time
a: on 30 days credit
### End
```json
{
    "name": "{{gen 'name'}}",
    "destination": "{{gen 'destination'}}",
    "dates": "{{gen 'eta'}}",
    "quantity": "{{gen 'quantity'}}"
    "grade": "{{#select 'grade'}}VLSFO{{or}}MGO{{or}}HFO{{/select}}",
    "terms": "{{gen 'terms'}}"
}
```
""".strip())()
    assert out['name'] == "Olympus"
    assert out['quantity'] == "1000mt"

    # out = guidance("""The height of the Sears tower is {{gen 'answer' pattern='[0-9]{0,10}'}} feet.""")()
    # assert 1000 < int(out['answer']) < 10000

    # out = guidance("""The Sun is very {{#select 'answer'}}hot{{or}}cold{{/select}}.""")()
    # assert out["answer"] == 'hot'

    # out = guidance("""The North Pole is {{#select 'answer'}}scorching{{or}}freezing{{/select}}.""")()
    # assert out["answer"] == 'freezing'