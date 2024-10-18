import pytest
import torch.nn.functional as F

from vllm.assets.image import ImageAsset
from ....conftest import IMAGE_ASSETS
from ..utils import check_embeddings_close



HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign":
    ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
     "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    #  "Select the portion of the image that isolates the object of the "
    #  "given label: The label of the object is stop sign<|im_end|>\n"
     "What is shown in this image?<|im_end|>\n"
     "<|im_start|>assistant\n<|endoftext|>"),
    "cherry_blossom":
    ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
     "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
     "What is shown in this image?<|im_end|>\n"
    #  "Represent the given image with the following question: What is in the image<|im_end|>\n" # noqa: E501
     "<|im_start|>assistant\n<|endoftext|>"),
})


MODELS = ["MrLight/dse-qwen2-2b-mrl-v1"]

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image
import os
import torch
class QwenVLEncoder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        attn = "flash_attention_2" if self.device == "cuda" else None

        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        self.processor = AutoProcessor.from_pretrained("MrLight/dse-qwen2-2b-mrl-v1")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "MrLight/dse-qwen2-2b-mrl-v1", 
            attn_implementation=attn, 
            torch_dtype=torch.bfloat16
        ).to(self.device).eval()
        self.processor.tokenizer.padding_side = "left"
        self.model.padding_side = "left"
        self.base_embed_dim = 1536

    def _get_embedding(self, last_hidden_state: torch.Tensor, dimension: int) -> torch.Tensor:
        reps = last_hidden_state[:, -1]
        reps = torch.nn.functional.normalize(reps[0, :dimension], p=2, dim=-1)
        return reps

    def embed(self, inp: dict, embed_dim: int = None) -> torch.Tensor:
        """
        inp: dict
            {
                "dtype": "image",
                "image": PIL.Image,
            }
            or 
            {
                "dtype": "text",
                "question": (str) the question to embed,
            }
        embed_dim: int
            Will slice embeddings like emb[:embed_dim]
        """
        if inp["dtype"] == "image":
            messages = [[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": inp["image"]},
                        {"type": "text", "text": "What is shown in this image?"}
                    ]
                }
            ]]
        else:
            messages = [[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": Image.new("RGB", (28, 28)), "resized_height":1 , "resized_width":1}, # need a dummy image here for an easier process.
                        {"type": "text", "text": f"Query: {inp['question']}"},
                    ]
                }
            ]]
        image_inputs, video_inputs = process_vision_info(messages)

        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) + "<|endoftext|>"
            for msg in messages
        ]
        inputs = self.processor(text=texts, images=image_inputs, padding="longest", return_tensors="pt").to(self.device)
        inputs = self.model.prepare_inputs_for_generation(**inputs, use_cache=False)
        # print(f'hf inputs:\n{inputs}')
        # print(f'len hf inputs: {len(inputs["input_ids"][0])}')

        with torch.no_grad():
            output = self.model(**inputs, return_dict=True, output_hidden_states=True)

        embeddings = self._get_embedding(output.hidden_states[-1], embed_dim)
        return embeddings


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:


    """
    Warning, have PYTHONPATH='~/qwen_vllm/vllm' otherwise it will use the other installed version
    """

    example_prompts = HF_IMAGE_PROMPTS
    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).
    images = [
        ImageAsset("stop_sign").pil_image.convert("RGB"),
        ImageAsset("cherry_blossom").pil_image.convert("RGB"),
    ]
    prompts = [
        {
            "prompt": prompt,
            "multi_modal_data": {
                "image": image,
            }
        } for prompt, image in zip(example_prompts, images)
    ]
    with vllm_runner(model,
                     max_model_len=4096,
                     max_num_seqs=2,
                     dtype=dtype,
                     enforce_eager=True) as vllm_model:
        vllm_outputs = vllm_model.encode(prompts)
    # with hf_runner(model, auto_cls=Qwen2VLForConditionalGeneration, dtype=dtype) as hf_model:
    #     all_inputs = hf_model.get_inputs(example_prompts)
    #     breakpoint()

    #     all_outputs = []
    #     for inputs in all_inputs:
    #         # Based on: https://github.com/TIGER-AI-Lab/VLM2Vec/blob/db3b951bccabba220c1f53ab46a734e50dd2fc08/src/model.py
    #         outputs = hf_model.model(
    #             **hf_model.wrap_device(inputs,
    #                                    device=hf_model.model.device.type),
    #             return_dict=True,
    #             output_hidden_states=True,
    #         )
    #         last_hidden_state = outputs.hidden_states[-1][0]
    #         reps = last_hidden_state[inputs.attention_mask[0].sum() - 1]
    #         pooled_output = F.normalize(reps, p=2, dim=-1)

    #         all_outputs.append(pooled_output.tolist())

    #     hf_outputs = all_outputs
    hf_model = QwenVLEncoder()
    hf_outputs = []
    for prompt, image in zip(example_prompts, images):
        # I am not actually using the prompts from above, I am only testing the prompt that is on HF
        # this is temporary, until I can write proper tests with their code
        inp = {"dtype": "image", "image": image}
        hf_outputs.append(hf_model.embed(inp).tolist())

    print(f'{vllm_outputs[0][:10]}')
    # print(f'Tensor diff')
    # print(f'{(torch.tensor(vllm_outputs[0]) - torch.tensor(hf_outputs[0])).abs().mean()}')
    # print(f'{(torch.tensor(vllm_outputs[1]) - torch.tensor(hf_outputs[1])).abs().mean()}')
    # exit()
    check_embeddings_close(
        embeddings_0_lst=hf_outputs,
        embeddings_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
