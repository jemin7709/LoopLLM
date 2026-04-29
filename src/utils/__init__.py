
from .opt_utils import load_model_and_tokenizer
from .opt_utils import get_loss
from .opt_utils import get_gradients
from .opt_utils import get_all_losses
from .opt_utils import sample_control
from .opt_utils import get_filtered_cands
from .opt_utils import is_entropy_low

from .string_utils import read_data
from .string_utils import get_chat_prompt
from .string_utils import generate_str
from .string_utils import generate_str_vllm
from .string_utils import get_nonascii_toks
from .string_utils import test_suffix
from .string_utils import test_suffix_vllm
from .string_utils import SuffixManager

MODEL_PATHS = {
    "llama2-13b": "meta-llama/Llama-2-13b-chat-hf",  # meta-llama/Llama-2-13b-chat-hf
    "glm4-9b": "zai-org/glm-4-9b-chat-hf",  # zai-org/glm-4-9b-chat-hf
    "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",  # meta-llama/Llama-2-7b-chat-hf
    "llama3-8b": "meta-llama/Llama-3.1-8B-Instruct",  # meta-llama/Llama-3.1-8B-Instruct
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",  # mistralai/Mistral-7B-Instruct-v0.3
    "falcon3-7b": "tiiuae/Falcon3-7B-Instruct",  # tiiuae/Falcon3-7B-Instruct
    "vicuna-7b": "lmsys/vicuna-7b-v1.5",  # lmsys/vicuna-7b-v1.5
    "phi4-mini": "microsoft/Phi-3.5-mini-instruct",  # microsoft/Phi-3.5-mini-instruct
    "llama3-3b": "meta-llama/Llama-3.2-3B-Instruct",  # meta-llama/Llama-3.2-3B-Instruct
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",  # Qwen/Qwen2.5-3B-Instruct
    "stablelm-3b": "stabilityai/stablelm-zephyr-3b",  # stabilityai/stablelm-zephyr-3b
    "gemma2-2b": "google/gemma-2-2b-it",  # google/gemma-2-2b-it
    "llama3-1b": "meta-llama/Llama-3.2-1B-Instruct",  # meta-llama/Llama-3.2-1B-Instruct
    "solar-open": "upstage/Solar-Open-100B",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
}
