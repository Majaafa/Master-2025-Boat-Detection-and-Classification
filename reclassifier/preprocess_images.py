import cv2
import numpy as np
import os
from tqdm import tqdm

def yolo_to_pixel_coordinates(yolo_bounding_box, image_width, image_height, padding_factor):
    x_center,y_center,bb_width,bb_height = map(float,yolo_bounding_box)

    x_start = (x_center-bb_width/2) * image_width
    x_end = (x_center+bb_width/2) * image_width
    y_start = (y_center-bb_height/2) * image_height
    y_end = (y_center+bb_height/2) * image_height

    padding_x = bb_width * image_width * padding_factor
    padding_y = bb_height * image_height * padding_factor

    if x_start < padding_x:
        x_start = 0
    else:
        x_start -= padding_x
    
    if y_start < padding_y:
        y_start = 0
    else:
        y_start -= padding_y

    return int(x_start), int(x_end + padding_x),int(y_start), int(y_end + padding_y)

def pad_image(cropped_image, padding_type):
    height, width = cropped_image.shape[:2]
    if height > width:
        left_pad = (height - width) // 2
        right_pad = height - width - left_pad

        if padding_type == "lb":
            padded_image = cv2.copyMakeBorder(cropped_image, 0, 0, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(0,0,0))
        elif padding_type == "erp":
            padded_image = cv2.copyMakeBorder(cropped_image, 0, 0, left_pad, right_pad, borderType=cv2.BORDER_REPLICATE)
        elif padding_type == "stack" and height > width*2:
            stack_numb = int(height / width)
            padded_image = np.hstack([cropped_image] * stack_numb)

    else:
        top_pad = (width - height) // 2
        bottom_pad = width - height - top_pad

        if padding_type == "lb":
            padded_image = cv2.copyMakeBorder(cropped_image, top_pad, bottom_pad, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
        elif padding_type == "erp":
            padded_image = cv2.copyMakeBorder(cropped_image, top_pad, bottom_pad, 0, 0, borderType=cv2.BORDER_REPLICATE)
        elif padding_type == "stack" and width > height*2:
            stack_numb = int(width / height)
            padded_image = np.vstack([cropped_image] * stack_numb)
    
    return padded_image

def square_image_crop(image, cropped_image,bb,blur_state = False):

    cropped_height, cropped_width = cropped_image.shape[:2]
    image_height, image_width = image.shape[:2]
    largest_side = max(cropped_height, cropped_width)
    x1,x2,y1,y2 = bb

    new_x_start = int(max(0, (x1+x2)//2 - largest_side // 2))
    new_y_start = int(max(0, (y1+y2)//2 - largest_side // 2))
    new_x_end = int(min(image_width, new_x_start + largest_side))
    new_y_end = int(min(image_height, new_y_start + largest_side))

    if new_x_end - new_x_start < largest_side:
        new_x_start = max(0, new_x_end - largest_side)
    if new_y_end - new_y_start < largest_side:
        new_y_start = max(0, new_y_end - largest_side)


    square_image_crop = image[new_y_start:new_y_end, new_x_start:new_x_end].copy()
    
    pad_offset_x1 = x1- new_x_start
    pad_offset_y1 = y1- new_y_start 
    pad_offset_x2 = x2 - new_x_start
    pad_offset_y2 = y2 - new_y_start

    if blur_state:
        blurred_square_image_crop = cv2.GaussianBlur(square_image_crop, (25, 25), 0)
        blurred_square_image_crop[pad_offset_y1:pad_offset_y2, pad_offset_x1:pad_offset_x2] = image[y1:y2, x1:x2]
        return blurred_square_image_crop
    
    square_image_crop[pad_offset_y1:pad_offset_y2, pad_offset_x1:pad_offset_x2] = image[y1:y2, x1:x2]
    return square_image_crop




def preprocess_images(image_folder, annotation_folder, output_image_folder,padding_type=None, padding_factor=0.0,blur_state=False):
    os.makedirs(output_image_folder, exist_ok=True)

    for image in tqdm(os.listdir(image_folder),desc="Processing images...."):
        image_file_path = os.path.join(image_folder, image)
        annotation_file_path = os.path.join(annotation_folder, os.path.splitext(image)[0]+".txt")

        image_cv2 = cv2.imread(image_file_path)
        image_height, image_width = image_cv2.shape[:2]

        with open(annotation_file_path, 'r') as f:
            annotations = f.readlines()

        for row, annotation in enumerate(annotations):
            the_annotation = annotation.strip().split()
            class_id = the_annotation[0]
            bounding_box = the_annotation[1:]
            x_start_padded, x_end_padded, y_start_padded, y_end_padded = yolo_to_pixel_coordinates(bounding_box, image_width, image_height, padding_factor)
            cropped_out_bb_image = image_cv2[y_start_padded:y_end_padded, x_start_padded:x_end_padded]

            if padding_type is not None:
                square_image = square_image_crop(image_cv2, cropped_out_bb_image, (x_start_padded, x_end_padded, y_start_padded, y_end_padded), blur_state)
                square_image_height, square_image_width = square_image.shape[:2]
                if square_image_height != square_image_width:
                    padded_image = pad_image(square_image, padding_type)
                else:
                    padded_image = square_image
                
                


            cropped_out_bb_image_filename = f"{os.path.splitext(image)[0]}_{row}.jpg"
            cropped_out_bb_image_filepath = os.path.join(output_image_folder, cropped_out_bb_image_filename)
            cv2.imwrite(cropped_out_bb_image_filepath, padded_image)



