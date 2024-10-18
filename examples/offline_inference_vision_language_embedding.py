from vllm import LLM
from vllm.assets.image import ImageAsset

image = ImageAsset("cherry_blossom").pil_image.convert("RGB")
# prompt = "<|image_1|> Represent the given image with the following question: What is in the image"  # noqa: E501
prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
          "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
          "What is shown in this image?<|im_end|>\n"
          "<|im_start|>assistant\n")

# Create an LLM.
llm = LLM(
    # model="TIGER-Lab/VLM2Vec-Full",
    model="MrLight/dse-qwen2-2b-mrl-v1",
    # model="Qwen/Qwen2-VL-2B-Instruct",
    trust_remote_code=True,
    max_model_len=4096,
    max_num_seqs=2,
    # mm_processor_kwargs={"num_crops": 16},
)

# Generate embedding. The output is a list of EmbeddingRequestOutputs.
outputs = llm.encode({"prompt": prompt, "multi_modal_data": {"image": image}})
# outputs = llm.generate({"prompt": prompt, "multi_modal_data": {"image": image}})

# Print the outputs.
for output in outputs:
    print(output.outputs.embedding)  # list of 3072 floats
