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

    out = guidance("The following are requests for fuel pricing for maritime vessels as conversations.\nFollowing the conversation is the parsed data from the conversation for the fuel request.\n\n### Start\na: Our good lady Lodesta Capella is completing discharging at Mersin and will go to dry dock at Tuzla.\nCould you check the indicative prices for two options:\n1) alongside at Mersin\n2) Barge supply Istanbul-Tuzla range\n\n50 mts MGO\nb: No problem\nb: CIA or need some credit terms?\na: COD or upto 7dd\nb: ok\na: thanks\nb: What at her dates in Mersin?\na: ETC 23-25 Apr\na: ETA Tuzla 29-30 Apr\nb: tks\n### End\n```json\n{\n    \"name\": \"Lady Lodesta Capella\",\n    \"destination\": \"Mersin\",\n    \"dates\": \"Apr 23-25\",\n    \"quantity\": \"1000mt\"\n    \"grade\": \"MGO\",\n    \"terms\": \"COD or up to 7dd\"\n}\n```\n\n### Start\na: looking to get a quote to refuel the olympus\na: needs 1000 tons vlsfo at ara in a weeks time\na: on 30 days credit\n### End\n```json\n{\n    \"name\": \"{{gen 'name'}}\",\n    \"destination\": \"{{gen 'destination'}}\",\n    \"dates\": \"{{gen 'eta'}}\",\n    \"quantity\": \"{{gen 'quantity'}}\"\n    \"grade\": \"{{#select 'grade'}}VLSFO{{or}}MGO{{or}}HFO{{/select}}\",\n    \"terms\": \"{{gen 'terms'}}\"\n}\n```")
    assert out['name'] == "Olympus"
    assert out['quantity'] == "1000mt"

    # out = guidance("""The height of the Sears tower is {{gen 'answer' pattern='[0-9]{0,10}'}} feet.""")()
    # assert 1000 < int(out['answer']) < 10000

    # out = guidance("""The Sun is very {{#select 'answer'}}hot{{or}}cold{{/select}}.""")()
    # assert out["answer"] == 'hot'

    # out = guidance("""The North Pole is {{#select 'answer'}}scorching{{or}}freezing{{/select}}.""")()
    # assert out["answer"] == 'freezing'
