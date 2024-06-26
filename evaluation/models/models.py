import asyncio
import os

import evaluation.utils

fetched_model_configs = {}
fetched_model_configs_lock = asyncio.Lock()


async def fetch_model_config(model_name: str):
    await fetched_model_configs_lock.acquire()

    if model_name in fetched_model_configs:
        model_config = fetched_model_configs[model_name]
        fetched_model_configs_lock.release()
        return model_config

    # Check if the model is a llama_cpp model
    if "gguf" in model_name.lower():
        fetched_model_configs[model_name] = None
        fetched_model_configs_lock.release()
        return None
        
    import transformers

    # If model_name is a local file, use it directly
    if os.path.exists(model_name):
        model_config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        fetched_model_configs[model_name] = model_config
        fetched_model_configs_lock.release()
        return model_config

    # Default behavior: download from Hugging Face Hub
    try:
        model_config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        fetched_model_configs[model_name] = model_config
    except Exception as e:
        fetched_model_configs[model_name] = None
        fetched_model_configs_lock.release()
        raise e

    fetched_model_configs_lock.release()
    return model_config


async def get_dtype(model_name: str):
    import torch

    # Check if the model is a llama_cpp model
    if "gguf" in model_name.lower():
        return torch.float16  # or whatever default dtype is appropriate

    # Check if the model is a local file
    if os.path.exists(model_name):
        config = await fetch_model_config(model_name)
        if config:
            return config.torch_dtype
        else:
            return torch.float16  # or appropriate default dtype

    return (await fetch_model_config(model_name)).torch_dtype

async def get_supported_inference_backends(model_name: str):
    if "starchat" in model_name:
        # vLLM currently does not support starchat.
        # See https://github.com/vllm-project/vllm/issues/380
        return ["tgi", "hf_transformers"]

    generally_supported_model_types = [
        "llama",  # LLaMA & LLaMA-2
        "gpt_neox",  # EleutherAI Pythia models
        "gpt_bigcode",  # Starcoder
        "mpt",  # MPT models from MosaicML
        # All of these are some variant of falcon from TII
        "RefinedWeb",
        "RefinedWebModel",
        "falcon",
    ]

    model_type = (await fetch_model_config(model_name)).model_type
    if model_type in generally_supported_model_types:
        return ["vllm", "tgi", "hf_transformers"]
    if model_type in ["mistral"]:
        return ["vllm", "hf_transformers"]

    return []


def is_tgi_installed():
    return os.path.exists("text-generation-inference")


async def get_inference_backend(model_path: str):
    supported_backends = await get_supported_inference_backends(model_path)

    if "vllm" in supported_backends:
        return "vllm"

    if "tgi" in supported_backends:
        if is_tgi_installed():
            return "tgi"
        print(
            'WARNING: The model "'
            + model_path
            + '" can be greatly accelerated by text-generation-inference, but it is not installed.'
        )

    if "hf_transformers" in supported_backends:
        return "hf_transformers"

    raise Exception('No inference backend supported for model "' + model_path)




async def ensure_model_file(model_name: str):
    from huggingface_hub import hf_hub_download, list_repo_files, HfApi, login
    from requests.exceptions import HTTPError
    import os
    
    # Check if the model_name exists locally first
    if os.path.exists(model_name):
        return model_name

    # Split the model name into repo_id and optional filename
    parts = model_name.split("/")
    if len(parts) > 2:
        repo_id = "/".join(parts[:-1])
        filename = parts[-1]
    else:
        repo_id = model_name
        filename = None

    # Check if the filename is a GGUF file
    if filename and filename.lower().endswith(".gguf"):
        try:
            # Attempt to download the specified GGUF file
            model_name = hf_hub_download(repo_id=repo_id, filename=filename)
        except HTTPError as e:
            if e.response.status_code == 404:
                # If the specified GGUF file is not found, search for any GGUF file in the repo
                repo_files = list_repo_files(repo_id)
                for file in repo_files:
                    if file.lower().endswith(".gguf"):
                        model_name = hf_hub_download(repo_id=repo_id, filename=file)
                        break
                else:
                    raise ValueError(f"No GGUF file found in Hugging Face repository {repo_id}")
            elif e.response.status_code == 401:
                # Handle unauthorized error, possibly due to missing authentication
                token = os.getenv('HUGGINGFACE_TOKEN')
                if token:
                    login(token=token)
                    model_name = hf_hub_download(repo_id=repo_id, filename=filename)
                else:
                    raise ValueError(f"Unauthorized access to repository {repo_id}. Please ensure you have the correct access token.")
            else:
                raise e
    else:
        # Handle the case where only the repository is provided and it's a GGUF model
        try:
            # Attempt to download the first available GGUF file in the repository
            repo_files = list_repo_files(repo_id)
            for file in repo_files:
                if file.lower().endswith(".gguf"):
                    model_name = hf_hub_download(repo_id=repo_id, filename=file)
                    break
            else:
                raise ValueError(f"No GGUF file found in Hugging Face repository {repo_id}")
        except HTTPError as e:
            if e.response.status_code == 401:
                # Handle unauthorized error, possibly due to missing authentication
                token = os.getenv('HUGGINGFACE_TOKEN')
                if token:
                    login(token=token)
                    repo_files = list_repo_files(repo_id)
                    for file in repo_files:
                        if file.lower().endswith(".gguf"):
                            model_name = hf_hub_download(repo_id=repo_id, filename=file)
                            break
                    else:
                        raise ValueError(f"No GGUF file found in Hugging Face repository {repo_id}")
                else:
                    raise ValueError(f"Unauthorized access to repository {repo_id}. Please ensure you have the correct access token.")
            else:
                raise e

    return model_name

async def create_model(
    model_type: str, 
    model_name: str, 
    model_args: dict[str, str], 
    backend_params=None,  # Added
    **kwargs  
):
    from evaluation.models.alpaca_with_prefix import AlpacaWithPrefix
    from evaluation.models.alpaca_without_prefix import AlpacaWithoutPrefix
    from evaluation.models.chatml import ChatML
    from evaluation.models.debug import Debug
    from evaluation.models.dolphin import Dolphin
    from evaluation.models.falcon_instruct import FalconInstruct
    from evaluation.models.fastchat import Fastchat
    from evaluation.models.guanaco import Guanaco
    from evaluation.models.llama2_chat import Llama2Chat
    from evaluation.models.open_ai import OpenAI
    from evaluation.models.open_ai import OpenAIJudge
    from evaluation.models.open_assistant import OpenAssistant
    from evaluation.models.openchat_llama2_v1 import OpenchatLlama2V1
    from evaluation.models.stable_beluga import StableBeluga
    from evaluation.models.starchat import Starchat
    from evaluation.models.wizard_lm import WizardLM
    from evaluation.models.zephyr import Zephyr
    from evaluation.models.mistral_instruct import MistralInstruct

    model_classes = {
        "debug": Debug,
        "openai": OpenAI,
        "openai_judge": OpenAIJudge,
        "fastchat": Fastchat,
        "open-assistant": OpenAssistant,
        "guanaco": Guanaco,
        "falcon-instruct": FalconInstruct,
        "alpaca-without-prefix": AlpacaWithoutPrefix,
        "alpaca-with-prefix": AlpacaWithPrefix,
        "chatml": ChatML,
        "starchat": Starchat,
        "llama2-chat": Llama2Chat,
        "stable-beluga": StableBeluga,
        "dolphin": Dolphin,
        "openchat-llama2-v1": OpenchatLlama2V1,
        "wizard-lm": WizardLM,
        "zephyr": Zephyr,
        "mistral-instruct": MistralInstruct,
        #"llama_cpp": LlamaCpp,  # we do this differently now
    }

    if model_type not in model_classes:
        raise Exception('Unknown model type "' + model_type + '"')

    model_class = model_classes[model_type]
    model = model_class()    

    # Ensure the model file is present
    if "gguf" in model_name.lower():
        model_name = await ensure_model_file(model_name)
    
    # Conditionally pass backend_params only if the model requires it
    model_init_args = {**model_args, **kwargs}
    if backend_params is not None:
        model_init_args['backend_params'] = backend_params

    await model.init(model_name, **model_init_args)
    return model

async def compute_model_replies(model, conversations, *, progress_bar_description=None):
    if len(conversations) == 0:
        return []

    end_token = "<|im_end|>" ## temporary quick & dirty fix for the moment, somehow this did not work otherwise, maybe fixable by other start parameters

    async def compute_reply(conversation):
        if isinstance(conversation, list):
            reply = await model.reply(conversation)
        elif isinstance(conversation, dict):
            reply = await model.reply(**conversation)
        else:
            raise ValueError("Invalid conversation format")
        
        # Strip the end token if it exists at the end of the reply
        if reply.endswith(end_token):
            reply = reply[:-len(end_token)].rstrip()
        
        return reply
        
    return await evaluation.utils.process_with_progress_bar(
        items=conversations,
        process_fn=compute_reply,
        progress_bar_description=progress_bar_description,
    )


async def switch_inference_backend(new_inference_backend):
    import evaluation.models.fastchat
    import evaluation.models.huggingface_backends.hf_transformers
    import evaluation.models.huggingface_backends.tgi
    import evaluation.models.huggingface_backends.vllm_backend
    import evaluation.models.huggingface_backends.llama_cpp_backend  # new import

    unload_backend_fns = {
        "hf_transformers": evaluation.models.huggingface_backends.hf_transformers.unload_model,
        "vllm": evaluation.models.huggingface_backends.vllm_backend.unload_model,
        "tgi": evaluation.models.huggingface_backends.tgi.unload_model,
        "fastchat": evaluation.models.fastchat.unload_model,
        "llama_cpp": evaluation.models.huggingface_backends.llama_cpp_backend.unload_model,  # new reference
    }

    for inference_backend_name, unload_backend_fn in unload_backend_fns.items():
        if inference_backend_name == new_inference_backend:
            continue
        await unload_backend_fn()



async def unload_model():
    await switch_inference_backend(None)
