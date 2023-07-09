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

    # just make sure it runs
    out = guidance("""The height of the Sears tower is {{#select 'answer'}}unknown{{or}}very tall{{or}}very short{{/select}}""")()
    assert out['answer'] == 'tall'
    # assert int(out['answer']) > 1000

    out = guidance("""The Sun is very {{#select 'answer'}}hot{{or}}cold{{/select}}.""")()
    assert out["answer"] == 'hot'

    out = guidance("""The North Pole is {{#select 'answer'}}scorching{{or}}freezing{{/select}}.""")()
    assert out["answer"] == 'freezing'
