import guidance
import os
import glob
import pytest

from exllama_lib.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama_lib.tokenizer import ExLlamaTokenizer
from exllama_lib.generator import ExLlamaGenerator


def test_basic():
    model_directory =  "/root/llama-7b-gptq-4bit-128g"
    tokenizer_path = os.path.join(model_directory, "tokenizer.model")
    model_config_path = os.path.join(model_directory, "config.json")
    st_pattern = os.path.join(model_directory, "*.safetensors")
    model_path = glob.glob(st_pattern)[0]

    config = ExLlamaConfig(model_config_path)               # create config from config.json
    config.model_path = model_path                          # supply path to model weights file

    model = ExLlama(config)                                 # create ExLlama instance and load the weights
    tokenizer = ExLlamaTokenizer(tokenizer_path)            # create tokenizer from tokenizer model file

    cache = ExLlamaCache(model)                             # create cache for inference
    generator = ExLlamaGenerator(model, tokenizer, cache)   # create generator

    guidance.llm = guidance.llms.ExLLaMA(model=model, generator=generator, tokenizer=tokenizer, caching=False)

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
