from preprocess_images import preprocess_images
from reclassify import EN7_classify



preprocess_images(
    image_folder="test_data/images",
    annotation_folder="test_data/grounding_dino/yolo_labels_grounding_dino",
    output_image_folder="test_data/cropped_out_bb_02_gaus_erp",
    padding_type="erp",  
    padding_factor=0.2,
    blur_state=True
)

EN7_classify(
    preprocessed_images_folder="test_data/cropped_out_bb_02_gaus_erp",
    original_images_folder="test_data/images",
    auto_labels_folder="test_data/grounding_dino/yolo_labels_grounding_dino",
    output_labels_folder="test_data/run_1/EN7_reclassifier_output_labels",
    output_conf_score_txt_path="test_data/run_1/EN7_reclassifier_conf_scores.txt",
    output_not_classified_images_folder="test_data/run_1/EN7_reclassifier_not_classified_images",
    cache_path="test_data/run_1/EN7_reclassifier_cache.json",
    filter_state=True
)