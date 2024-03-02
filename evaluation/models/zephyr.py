from .huggingface import Huggingface


class Zephyr(Huggingface):
    def __init__(self, model_path, *, default_system_message="", **kwargs):
        super().__init__(
            model_path,
            user="<|user|>\n",
            assistant="<|assistant|>\n",
            system="<|system|>\n",
            default_system=default_system_message,
            end="</s>\n",
            **kwargs,
        )
