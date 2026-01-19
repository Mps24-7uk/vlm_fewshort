from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from PIL import Image
import torch

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

# load the vision-language model and processor
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    dtype="auto",
    device_map="auto"
)
model.eval()

def generate_chip_description(image_path, retrieved_matches):
    # build reference context from similarity matches
    reference_text = "\n".join(
        f"- {m['description']}" for m in retrieved_matches
    )

    # user prompt with structure + references
    prompt_text = f"""
Here are similar examples from the database:
{reference_text}

Now, based only on the given image and these examples, provide a concise,
factual, structured description of the observable physical characteristics
of the chip in the image (shape, orientation, body, caps/pins, surface, context).
"""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path
                },
                {
                    "type": "text",
                    "text": prompt_text
                }
            ]
        }
    ]

    # tokenize using the special chat template
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # generate output tokens
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.2,
            do_sample=False
        )

    # decode only the generated part (skip input tokens)
    # processor.batch_decode expects trimmed ids
    input_len = inputs["input_ids"].shape[-1]
    generated_ids_trimmed = generated_ids[:, input_len:]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    return output_text[0]
