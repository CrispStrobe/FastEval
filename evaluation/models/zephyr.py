from .huggingface import Huggingface


class Zephyr(Huggingface):
    async def init(self, model_path, *, default_system_message="", **kwargs):
        await super().init(
            model_path,
            user="<|user|>\n",
            assistant="<|assistant|>",
            system="<|system|>\n",
            default_system=default_system_message,
            end="<|end|>\n",
            **kwargs,
        )
