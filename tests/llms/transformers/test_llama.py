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

    out = guidance("The following are requests for fuel pricing for maritime vessels as conversations.\nFollowing the conversation is the parsed data from the conversation for the fuel request.\n\n### Start\na: Our good lady Lodesta Capella is completing discharging at Mersin and will go to dry dock at Tuzla.\nCould you check the indicative prices for two options:\n1) alongside at Mersin\n2) Barge supply Istanbul-Tuzla range\n\n50 mts MGO\nb: No problem\nb: CIA or need some credit terms?\na: COD or upto 7dd\nb: ok\na: thanks\nb: What at her dates in Mersin?\na: ETC 23-25 Apr\na: ETA Tuzla 29-30 Apr\nb: tks\n### End\n```json\n{\n    \"name\": \"Lady Lodesta Capella\",\n    \"destination\": \"Mersin\",\n    \"dates\": \"Apr 23-25\",\n    \"quantity\": \"1000mt\"\n    \"grade\": \"MGO\",\n    \"terms\": \"COD or up to 7dd\"\n}\n```\n\n### Start\na: looking to get a quote to refuel the olympus\na: needs 1000 tons vlsfo at ara in a weeks time\na: on 30 days credit\n### End\n```json\n{\n    \"name\": \"{{gen 'name'}}\",\n    \"destination\": \"{{gen 'destination'}}\",\n    \"dates\": \"{{gen 'eta'}}\",\n    \"quantity\": \"{{gen 'quantity'}}\"\n    \"grade\": \"{{#select 'grade'}}VLSFO{{or}}MGO{{or}}HFO{{/select}}\",\n    \"terms\": \"{{gen 'terms'}}\"\n}\n```")
    assert out['name'] == "Olympus"
    assert out['quantity'] == "1000mt"

    # out = guidance("""The height of the Sears tower is {{gen 'answer' pattern='[0-9]{0,10}'}} feet.""")()
    # assert 1000 < int(out['answer']) < 10000

    # out = guidance("""The Sun is very {{#select 'answer'}}hot{{or}}cold{{/select}}.""")()
    # assert out["answer"] == 'hot'

    # out = guidance("""The North Pole is {{#select 'answer'}}scorching{{or}}freezing{{/select}}.""")()
    # assert out["answer"] == 'freezing'