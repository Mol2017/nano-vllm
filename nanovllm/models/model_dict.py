from nanovllm.models.qwen2 import Qwen2ForCausalLM
from nanovllm.models.qwen3 import Qwen3ForCausalLM

model_dict = {
    "qwen2": Qwen2ForCausalLM,
    "qwen3": Qwen3ForCausalLM,
}