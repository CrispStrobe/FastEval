import asyncio
import queue
import uuid

from evaluation.models.huggingface_backends.data_parallel import DataParallelBackend


async def create_model(*, model_path, tokenizer_path, dtype, backend_params=None):
    import torch
    import vllm

    # Process backend_params
    engine_args = vllm.AsyncEngineArgs(
        model=model_path,
        tokenizer=tokenizer_path,
        tensor_parallel_size=torch.cuda.device_count(),
        dtype=str(dtype).replace("torch.", ""),
        disable_log_requests=True,
        trust_remote_code=True,
        max_num_seqs=1024,
        max_model_len=8192,
    )

    if backend_params:
        for key, value in backend_params.items():
            if hasattr(engine_args, key):
                setattr(engine_args, key, value)
            else:
                raise AttributeError(f"Parsing of backend engine parameters failed: enginge_args has no attribute '{key}'")

    engine = vllm.AsyncLLMEngine.from_engine_args(
        engine_args,
        start_engine_loop=False,
    )

    return {
        "tokenizer_path": tokenizer_path,
        "model_path": model_path,
        "dtype": dtype,
        "engine": engine,
    }


async def compute_model_response(*, model, item):
    import vllm

    prompt = item["prompt"]
    temperature = item["temperature"]
    max_new_tokens = item["max_new_tokens"]

    if temperature is None:
        temperature = 1.0

    if isinstance(prompt, tuple):
        if prompt[0] != "tokens":
            raise Exception("Unknown prompt type")
        args = {"prompt_token_ids": prompt[1], "prompt": None}
    else:
        args = {"prompt": prompt}

    if not model['engine'].is_running:
        model['engine'].start_background_loop()

    response_generator = model["engine"].generate(
        **args,
        sampling_params=vllm.SamplingParams(
            # See https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
            best_of=None,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            temperature=temperature,
            top_p=1.0,
            top_k=-1,
            use_beam_search=False,
            max_tokens=max_new_tokens,
        ),
        request_id=uuid.uuid4()
    )

    response = None
    async for response_part in response_generator:
        if not response_part.finished:
            continue
        assert response is None
        outputs = response_part.outputs
        assert len(outputs) == 1
        response = outputs[0].text

    return response


backend = DataParallelBackend(
    backend_name="vllm",
    worker_functions={
        "create_model": create_model,
        "compute_model_response": compute_model_response,
    },
    worker_is_blocking=False,
)

def filter_inference_kwargs(kwargs):
    allowed_keys = {"prompt", "tokenizer_path", "model_path", "dtype", "max_new_tokens", "temperature"}
    return {k: v for k, v in kwargs.items() if k in allowed_keys}

#async def run_inference(**kwargs):
#    return await backend.run_inference(**kwargs, max_batch_size=1)
async def run_inference(**kwargs):
    filtered_kwargs = filter_inference_kwargs(kwargs)  # Filter the kwargs to remove unexpected arguments
    item = {
        "messages": kwargs.get("messages", [{"role": "system", "content": "No previous messages"}]),  # Ensure messages are passed
    }

    # Include backend_params if provided
    if "backend_params" in kwargs:
        filtered_kwargs["backend_params"] = kwargs["backend_params"]

    return await backend.run_inference(**filtered_kwargs, item=item, max_batch_size=1)

async def unload_model():
    return await backend.unload_model()
