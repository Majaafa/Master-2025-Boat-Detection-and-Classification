# Trained with 1000 good and 1000 bad synthetic images

from ultralytics import YOLO

if __name__ == "__main__":
  
    model = YOLO('yolo11x-cls.pt')

    model.train(
        data=r'c:\Users\halvo\OneDrive\Skrivebord\good_bad_yolo',
        epochs=20,
        imgsz=320,
        batch=32,
        lr0=1e-4,
        warmup_epochs=10,
        dropout=0.4,
        patience=100,
        optimizer='AdamW',
        project='yolo_class_train',
        name='yolo11_classification_good_bad',

        # Augmentations
        hsv_h=0.07,
        hsv_s=0.4,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.3,
        scale=0.3,
        shear=8.0,
        perspective=0.0002,
        fliplr=0.4,
        mosaic=0.6,
        mixup=0.1,
        copy_paste=0.05,
        erasing=0.2,

        augment=True
    )
