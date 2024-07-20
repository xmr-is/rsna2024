import albumentations as A

class Augmentation(object):

    def transform_train(train_data):
        transforms_train = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=aug_prob),
            A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5),
                A.GaussNoise(var_limit=(5.0, 30.0)),
            ], p=aug_prob),

            A.OneOf([
                A.OpticalDistortion(distort_limit=1.0),
                A.GridDistortion(num_steps=5, distort_limit=1.),
                A.ElasticTransform(alpha=3),
            ], p=aug_prob),

            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=aug_prob),
            A.Resize(image_size, image_size),
            A.CoarseDropout(max_holes=16, max_height=64, max_width=64, min_holes=1, min_height=8, min_width=8, p=aug_prob),    
            A.Normalize(mean=0.5, std=0.5)
        ])
        return transforms_train

    def transform_valid(valid_data):
        transforms_valid = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=0.5, std=0.5)
        ])
        return transforms_valid

    def debug():
        if not not_debug or not apply_aug:
            transforms_train = transforms_val
        return transforms_train
