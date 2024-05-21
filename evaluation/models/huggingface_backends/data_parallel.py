import asyncio
from llama_cpp import Llama
from evaluation.models.huggingface_backends.data_parallel import DataParallelBackend

async def create_model(*, model_path, dtype, chat_format=None, verbose=False, tokenizer_path=None):
    llm_args = {
        "model_path": model_path,
        "n_ctx": 2048,
        "n_gpu_layers": 100,

        #"chat_format": "chatml",
    }
    #if chat_format is not None:
    #    llm_args["chat_format"] = chat_format
    if verbose:
        llm_args["verbose"] = verbose

    llm = Llama(**llm_args)

    return {
        "model": llm,
    }

async def compute_model_response(*, model, item):
    print ("working on item: ", item)
    llm = model["model"]
    messages = item.get("messages", [])

    prompt = item["prompt"]
    messages = [{"role": "user", "content": prompt}]

    print ("\nmessages: ", messages)
    response = llm.create_chat_completion(
            messages=messages,
            #max_new_tokens=item.get("max_new_tokens", 1024),
            #temperature=item.get("temperature", 0.7)
    )
    print ("\ncomputed: ", response)
    print ("\nreturned answer: ", response["choices"][0]["message"]["content"])
    return response["choices"][0]["message"]["content"]

def filter_inference_kwargs(kwargs):
    allowed_keys = {"model_path", "dtype", "max_new_tokens", "temperature", "chat_format", "verbose", "prompt", "messages"}
    return {k: v for k, v in kwargs.items() if k in allowed_keys}

backend = DataParallelBackend(
    backend_name="llama_cpp",
    worker_functions={
        "create_model": create_model,
        "compute_model_response": compute_model_response,
    },
    worker_is_blocking=False,
)

async def run_inference(**kwargs):
    filtered_kwargs = filter_inference_kwargs(kwargs)  # Filter the kwargs to remove unexpected arguments
    item = {
        "messages": kwargs.get("messages", [{"role": "system", "content": ""}]),  # Ensure messages are passed
    }
    return await backend.run_inference(**filtered_kwargs, item=item, max_batch_size=1)

async def unload_model():
    return await backend.unload_model()
