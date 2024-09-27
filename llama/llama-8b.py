import ray
from ray import serve
import transformers
import torch
import os

HF_TOKEN = os.getenv("HF_TOKEN")

# Define the Llama model class for Ray Serve
@serve.deployment(num_replicas=2, ray_actor_options={"num_gpus": 1})
class LlamaModel:
    def __init__(self):
        model_name = "meta-llama/Meta-Llama-3.1-8B"
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)

        # with init_empty_weights():
        #     self.model = AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN)

        self.pipeline = transformers.pipeline("text-generation", model=model_name, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto", token=HF_TOKEN)

        # device_map = infer_auto_device_map(self.model, no_split_module_classes=["LlamaBlock"])
        # self.model = self.model.from_pretrained(model_name, device_map=device_map, token=HF_TOKEN)

    def __call__(self, request: dict) -> dict:
        input_text = request["text"]
        # inputs = self.tokenizer(input_text, return_tensors="pt").to("cuda:0")  # Send input to first GPU
        # outputs = self.model.generate(inputs["input_ids"])
        # generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = self.pipeline(input_text)
        return {"generated_text": generated_text}

llama_deploy = LlamaModel.bind()