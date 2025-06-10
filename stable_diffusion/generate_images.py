# Generate images using Stable Diffusion with LoRA weights, with prompts from a text file.
# https://huggingface.co/stabilityai/stable-diffusion-2-1

import torch
from diffusers import StableDiffusionPipeline
from pathlib import Path

# Load the pipeline
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  
    variant="fp16"
)
pipe = pipe.to("cuda")

# Load LoRA weights
# lora_path = "pytorch_lora_weights.safetensors"
# pipe.load_lora_weights(".", weight_name=lora_path)

# loRA scaling depending on how strong you want the LoRA to influence the output
# pipe.fuse_lora(lora_scale=1.0)

# Read prompts from file
prompt_file = f"stable_diffusion/prompt_example.txt"
with open(prompt_file, "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

# Output directory
output_dir = Path(f"stable_diffusion/generated_images")
output_dir.mkdir(parents=True, exist_ok=True)

# Generate and save images
for idx, prompt in enumerate(prompts):
    print(f"Generating images for prompt {idx}/{len(prompts)}: {prompt}")
    images = pipe(prompt, num_inference_steps=50, num_images_per_prompt=2).images
    for i, img in enumerate(images):
        filename = output_dir / f"prompt{idx:02d}_img{i:02d}.png"
        img.save(filename)
