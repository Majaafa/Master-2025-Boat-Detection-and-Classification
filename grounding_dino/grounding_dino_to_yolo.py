# Runs Grounding DINO on all images, saves YOLO labels and (optional) visualizations

import os
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from torchvision.ops import nms
from functions import area_overlap_filter, convert_to_yolo, draw_detection, get_class_id, save_dino_score

# Grounding DINO pre-trained model choice
# https://huggingface.co/IDEA-Research/grounding-dino-base
model_id = "IDEA-Research/grounding-dino-base"

device = "cuda" if torch.cuda.is_available() else "cpu"

input_image_folder = "test_data/images"
output_label_folder = "test_data/grounding_dino/yolo_labels_grounding_dino"
output_viz_folder = "test_data/grounding_dino/vizualization_outputs"
save_visualization = True
output_score_path = f"test_data/grounding_dino/scores/{output_label_folder}_scores.txt"

confidence_threshold = 0.3
text_confidence_threshold = 0.3
area_overlap_threshold = 0.95
fallback_label = 1

# List of labels for Grdounding DINO to detect boats
text_labels = [
    "sail", "sailboat", "yawl", "catamaran", "trimaran", "schooner", "ketch", "pirateship",
    "kayak", "canoe", "rowboat", "rowingboat", "paddleboat", "gondola",
    "buoy", "plunger",
    "hobby", "hobbyboat", "cabincruiser", "motorboat", "speedboat",
    "barge", "ship", "containership", "liner", "ocean liner", "cargo ship", "freighter",
    "yacht", "lifeboat", "tugboat", "pilotboat", "fishingboat", "fireboat"
]

# Setup
os.makedirs(output_label_folder, exist_ok=True)
if save_visualization:
    os.makedirs(output_viz_folder, exist_ok=True)

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

image_files = [f for f in os.listdir(input_image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for filename in image_files:
    image_path = os.path.join(input_image_folder, filename)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # run Grounding DINO
    inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=confidence_threshold,
        text_threshold=text_confidence_threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    boxes = results["boxes"]
    scores = results["scores"]
    labels = results["text_labels"]

    # apply NMS or AOR (pick one)
    keep = area_overlap_filter(boxes, scores, area_overlap_threshold)
    # keep = nms(boxes, scores, area_overlap_threshold)

    boxes = boxes[keep]
    scores = scores[keep]
    labels = [labels[i] for i in keep]

    yolo_lines = []
    draw = ImageDraw.Draw(image, "RGBA") if save_visualization else None
    font = ImageFont.load_default() if save_visualization else None

    # go through final boxes and convert to YOLO format
    for idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        cls = get_class_id(label, fallback_class=fallback_label)
        yolo_lines.append(f"{cls} {convert_to_yolo(box.tolist(), width, height)}")

        # draw box on image if enabled
        if save_visualization:
            label_text = f"{label} [{cls}]: ({score:.2f})"
            draw_detection(draw, box.tolist(), label_text, idx, font)

    # create label file path
    base_name = os.path.splitext(filename)[0]
    label_path = os.path.join(output_label_folder, f"{base_name}.txt")
    
    # Write to the file
    with open(label_path, "w") as f:
        f.write("\n".join(yolo_lines))

    # save image with drawn boxes if enabled
    if save_visualization:
        viz_path = os.path.join(output_viz_folder, f"{base_name}.jpg")
        image.save(viz_path)

    print(f"Saved {len(yolo_lines)} boxes to {label_path}")

    # save confidence scores to file to be used for evaluation
    save_dino_score(scores, image_path, output_score_path)