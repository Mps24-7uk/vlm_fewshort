import torch
from transformers import SegGptImageProcessor, SegGptForImageSegmentation
from PIL import Image
import matplotlib.pyplot as plt

def load_image(path):
    """Load an image as PIL."""
    return Image.open(path).convert("RGB")

def run_seggpt_inference(
    image_path: str,
    prompt_image_path: str,
    prompt_mask_path: str,
    num_labels: int = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Perform one-shot segmentation with SegGPT.

    Args:
        image_path: Path to the target image to segment.
        prompt_image_path: Path to example image (prompt).
        prompt_mask_path: Path to segmentation mask for the prompt.
        num_labels: Number of classes (excluding background) in the prompt mask.
        device: "cuda" or "cpu".
    Returns:
        seg_mask: A segmentation mask (H x W) as a numpy array.
    """

    # Model ID on Hugging Face
    model_id = "BAAI/seggpt-vit-large"

    # Load processor and model
    image_processor = SegGptImageProcessor.from_pretrained(model_id)
    model = SegGptForImageSegmentation.from_pretrained(model_id)
    model.to(device).eval()

    # Load images and prompt mask
    image = load_image(image_path)
    prompt_image = load_image(prompt_image_path)
    prompt_mask = Image.open(prompt_mask_path)

    # Encode inputs
    inputs = image_processor(
        images=image,
        prompt_images=prompt_image,
        prompt_masks=prompt_mask,
        num_labels=num_labels,
        return_tensors="pt",
    ).to(device)

    # Inference (no gradients)
    with torch.no_grad():
        outputs = model(**inputs)

    # Postprocess to get a segmentation mask
    target_sizes = [image.size[::-1]]  # height,width for post_process
    seg_mask = image_processor.post_process_semantic_segmentation(
        outputs, target_sizes, num_labels=num_labels
    )[0]

    return seg_mask

def visualize_segmentation(image_path, seg_mask):
    """Display original image + segmentation overlay."""
    image = load_image(image_path)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.imshow(seg_mask, alpha=0.5, cmap="jet")
    plt.title("SegGPT Segmentation")
    plt.axis("off")

    plt.show()

# === Example Usage ===
if __name__ == "__main__":
    seg_mask = run_seggpt_inference(
        image_path="data/query.jpg",
        prompt_image_path="data/prompt_image.jpg",
        prompt_mask_path="data/prompt_mask.png",
        num_labels=10,  # number of classes in prompt mask
    )
    visualize_segmentation("data/query.jpg", seg_mask)
