#
import torch
from PIL import Image, ImageDraw, ImageFont
import os
from torchvision.ops import box_iou

def get_class_id(label, fallback_class):

    label_map = {
        # Sailing-related
        "sailboat": 0, "yawl": 0, "catamaran": 0, "trimaran": 0, "schooner": 0,
        "ketch": 0, "pirateship": 0, "sail": 0,

        # Small paddle boats
        "kayak": 1, "canoe": 1, "rowboat": 1, "rowingboat": 1, "paddleboat": 1, "gondola": 1,

        # Floating object
        "buoy": 2, "plunger": 2,

        # Small motor/recreational boats
        "hobby": 3, "hobbyboat": 3, "cabincruiser": 3, "motorboat": 3, "speedboat": 3,

        # Large working ships
        "barge": 4, "ship": 4, "containership": 4, "liner": 4,
        "ocean liner": 4, "cargo ship": 4, "freighter": 4,

        # Other boats
        "yacht": 5, "lifeboat": 5, "tugboat": 5, "pilotboat": 5,
        "fishingboat": 5, "fireboat": 5,
    }

    return label_map.get(label.lower(), fallback_class)


# AOR (Area Overlap Ratio) filter 
def area_overlap_filter(boxes, scores, threshold=0.8):

    keep = []
    suppressed = torch.zeros(len(scores), dtype=torch.bool, device=boxes.device)
    sorted_indices = scores.argsort(descending=True)


    for i in sorted_indices:

        if suppressed[i]:
            continue
        else:
            keep.append(i.item())
            box_i = boxes[i]

        # Calulate AOR for all other boxes and suppress if sufficiently overlapping
        for j in sorted_indices:

            if i == j or suppressed[j]:
                continue

            box_j = boxes[j]
            
            # Calculate intersection
            x1 = max(box_i[0], box_j[0])
            y1 = max(box_i[1], box_j[1])
            x2 = min(box_i[2], box_j[2])
            y2 = min(box_i[3], box_j[3])

            inter_w = max(0, x2 - x1)
            inter_h = max(0, y2 - y1)
            inter_area = inter_w * inter_h

            if inter_area == 0:
                continue

            # Calculate areas
            area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
            area_j = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])
            smaller_area = min(area_i, area_j)

            # Check overlap ratio
            area_overlap_ratio = inter_area / smaller_area
            
            if area_overlap_ratio >= threshold:
                suppressed[j] = True

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

def convert_to_yolo(box, image_w, image_h):
    x1, y1, x2, y2 = box
    xc = (x1 + x2) / 2 / image_w
    yc = (y1 + y2) / 2 / image_h
    w = (x2 - x1) / image_w
    h = (y2 - y1) / image_h
    return f"{xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"

def draw_detection(draw, box, label_text, idx, font):

    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 255), width=2)

    bbox = font.getbbox(label_text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    padding = 2
    text_x = x1 + 2
    text_y = y1 + 2

    draw.rectangle(
        [text_x - padding, text_y - padding,
         text_x + text_width + padding, text_y + text_height + padding],
        fill=(255, 0, 0, 200)
    )
    draw.text((text_x, text_y), label_text, fill="white", font=font)

    box_idx_text = f"{idx}"
    bbox = font.getbbox(box_idx_text)
    idx_text_width = bbox[2] - bbox[0]
    idx_text_height = bbox[3] - bbox[1]
    draw.text((x2 - idx_text_width - 2, y2 - idx_text_height - 2), box_idx_text, fill="white", font=font)

def save_dino_score(score,image_file,output_score_path):
    os.makedirs(os.path.dirname(output_score_path), exist_ok=True)
    with open(output_score_path, "a") as f:
        f.write(str(os.path.basename(image_file))+" "+str(score)+"\n")

