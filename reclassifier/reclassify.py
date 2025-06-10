import os
import torch
from torchvision import transforms
from PIL import Image
import os
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
import ast
import torch.nn.functional as F
from tqdm import tqdm
import shutil
import json
from collections import defaultdict

def get_classname_from_class_ids(class_ids, class_dict):
    class_names = []
    for class_id in class_ids:
        for key, value in class_dict.items():
            if key == class_id:
                class_names.append(value)
    return class_names

def get_class_dict(imagenet_classes):
    with open(imagenet_classes, "r") as f:
        class_dict = ast.literal_eval(f.read())
    return class_dict

def get_defined_class_name():
    sailboat = ["yawl","trimaran","catamaran","schooner","pirate, pirate ship","sailboat"]
    mediumboat = ["fireboat","lifeboat","mediumboat"]
    paddleboat = ["canoe","gondola","rowingboat","paddleboat","paddle, boat paddle"]
    bigship = ["container ship, containership, container vessel","liner, ocean liner","bigship","bigboat"]
    hobbyboat = ["speedboat","hobbyboat"]
    buoy = ["submarine, pigboat, sub, U-boat","buoy"]

    return sailboat, mediumboat, paddleboat, bigship, hobbyboat, buoy

def update_class_in_txt(original_images_folder, image_path, new_class_id, auto_label_folder_name, class_conf_prob,output_labels_folder, output_not_classified_images_folder,filter_on=True):
    os.makedirs(output_labels_folder, exist_ok=True)

    image_filename = os.path.basename(image_path)
    base_name_with_index, filetype = os.path.splitext(image_filename)  
    base_name, row_index = base_name_with_index.rsplit("_", 1)
    row_index = int(row_index)
    label_path = os.path.join(auto_label_folder_name, base_name + ".txt")

    output_path = os.path.join(output_labels_folder, os.path.basename(label_path))
    if not os.path.exists(output_path):
        print(f"Label file {output_path} does not exist. Skipping.")
        return
    
    with open(output_path, "r") as file:
        labels = file.readlines()
   
    label = labels[row_index].strip().split()

    if filter_on:
        if new_class_id == None:
            print(f"Image {image_filename} with confidence {class_conf_prob} is filtered out. Type: {new_class_id}")
            manual_check_folder_path =  output_not_classified_images_folder
            os.makedirs(manual_check_folder_path, exist_ok=True)
            images_manual_check_folder = os.path.join(manual_check_folder_path, "images")
            images_orig_path = os.path.join(original_images_folder, base_name + filetype)
            labels_manual_check_folder = os.path.join(manual_check_folder_path, "labels")
            
            os.makedirs(images_manual_check_folder, exist_ok=True)
            os.makedirs(labels_manual_check_folder, exist_ok=True)

            images_manual_check_path = os.path.join(images_manual_check_folder, base_name + filetype)
            labels_manual_check_path = os.path.join(labels_manual_check_folder, base_name + ".txt")

            if os.path.exists(images_orig_path):
                shutil.copy(images_orig_path,images_manual_check_path)
                shutil.move(output_path, labels_manual_check_path)
            return
        else:
            label[0] = str(new_class_id) 
            labels[row_index] = " ".join(label) + "\n"
            
    elif new_class_id != None:
        label[0] = str(new_class_id)  
        labels[row_index] = " ".join(label) + "\n"

    with open(output_path, "w") as file:
        file.writelines(labels)


def groupe_conf_scores_for_image_txt(txt_file_path,new_txt_path):
    grouped_scores = defaultdict(list)

    with open(txt_file_path, 'r') as f:
        for line in f:
            filename, conf_score = line.strip().split()
            base_filename = os.path.splitext(filename)[0] 
            parts = base_filename.split('_')
            filename_image = '_'.join(parts[:-1]) + '.jpg'#todo
            print("filename_image",filename_image)
            grouped_scores[filename_image].append(float(conf_score))

    with open(new_txt_path, 'w') as file:
        for filename_image, scores in grouped_scores.items():
            tensor_scores = torch.tensor(scores, device='cuda:0')
            file.write(f"{filename_image} {tensor_scores}\n")



def EN7_classify(preprocessed_images_folder, original_images_folder, auto_labels_folder,output_labels_folder,output_conf_score_txt_path,output_not_classified_images_folder,cache_path,filter_state=False, use_cache=True):
    if not os.path.exists(output_labels_folder):
        shutil.copytree(auto_labels_folder, output_labels_folder)
    os.makedirs(os.path.dirname(output_conf_score_txt_path), exist_ok=True)
    with open(output_conf_score_txt_path, 'w') as file:
        pass
    with open('reclassifier/the_EN_score_train_new.txt', 'w') as file:
        pass  

    if use_cache and os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            cached_predictions = json.load(f)
    else:
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = EfficientNet_B7_Weights.DEFAULT
        model = efficientnet_b7(weights=weights)
        model.eval()
        preprocess = transforms.Compose([
            transforms.Resize(600),
            transforms.CenterCrop(528),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),    
        ])
        cached_predictions = []
        class_dict = get_class_dict("reclassifier/imagenet_classes.txt")

        for image in tqdm(os.listdir(preprocessed_images_folder), desc="EfficientNet classification", unit="image"):
            image_path = os.path.join(preprocessed_images_folder, image)
            image = Image.open(image_path).convert('RGB')
            input_tensor = preprocess(image).unsqueeze(0)  
            
            with torch.no_grad():
                output = model(input_tensor)
                soft_out = F.softmax(output, dim=1)
                top5_prob, top5_class_ids = torch.topk(soft_out, 5)
                class_ids = []
                class_probs = []

                for i in range(top5_prob.size(1)):
                    class_ids.append(top5_class_ids[0][i].item())
                    class_probs.append(top5_prob[0][i].item())

                class_name = get_classname_from_class_ids(class_ids, class_dict)
                cached_predictions.append({
                    "image_path": image_path,
                    "class_names": class_name,  
                    "prob": class_probs
                })

        with open(cache_path, 'w') as f:
            json.dump(cached_predictions, f)
    
    



    sailboat, mediumboat, paddleboat, bigship, hobbyboat, buoy = get_defined_class_name()
    #boat_class_dict = get_class_dict("reclassifier/boat_imagenet_classes.txt")

    for prediction in cached_predictions:
        image_path = prediction["image_path"]
        class_names = prediction["class_names"]
        class_probs = prediction["prob"]
        
        new_class_label = None 
        for n,top in enumerate(class_names):
            if top in sailboat:
                new_class_label = 0
            elif top in paddleboat:
                new_class_label = 1
            # elif top in buoy:
                #     type = 2
            elif top in hobbyboat:
                new_class_label = 3
            elif top in bigship:
                new_class_label = 4
            elif top in mediumboat:
                new_class_label = 5

            if new_class_label is not None:
                update_class_in_txt(original_images_folder=original_images_folder,
                                    image_path=image_path, 
                                    new_class_id=new_class_label, 
                                    auto_label_folder_name=auto_labels_folder,
                                    class_conf_prob=class_probs[n],
                                    output_labels_folder=output_labels_folder,
                                    output_not_classified_images_folder=output_not_classified_images_folder,
                                    filter_on=filter_state)
                with open("reclassifier/the_EN_score_train_new.txt", "a") as f:
                    f.write(f"{os.path.basename(image_path)} {class_probs[n]}\n")  
                break

        if new_class_label is None:
            update_class_in_txt(original_images_folder=original_images_folder,
                             image_path=image_path,
                             new_class_id=new_class_label,
                             auto_label_folder_name=auto_labels_folder,
                             class_conf_prob=0,
                             output_labels_folder=output_labels_folder,
                             output_not_classified_images_folder=output_not_classified_images_folder,
                             filter_on=filter_state)
            
            with open("reclassifier/the_EN_score_train_new.txt", "a") as f:
                    f.write(f"{os.path.basename(image_path)} {0.0}\n")

    groupe_conf_scores_for_image_txt("reclassifier/the_EN_score_train_new.txt",output_conf_score_txt_path)

