import os
from PIL import Image
from torchvision import transforms


augmentation_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(30, fill=(1,)), # Rotate up to 30 degrees with white corners
    transforms.RandomResizedCrop(256, scale=(0.9, 1.1)), # Zoom in/out
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=1),  # Smaller white patches
])

source_folder = "flowers"
augmented_folder = "flowers_augmented"

categories = ["train", "valid"]
for category in categories:
    category_path = os.path.join(source_folder, category)
    class_names = os.listdir(category_path)

    for class_name in class_names:
        augmented_class_path = os.path.join(augmented_folder, category, class_name)
        os.makedirs(augmented_class_path, exist_ok=True)

def augment_and_save_images(source_path, target_path, num_augmented=3):
    for category in categories:
        category_path = os.path.join(source_path, category)
        class_names = os.listdir(category_path)

        for class_name in class_names:
            class_path = os.path.join(category_path, class_name)
            images = os.listdir(class_path)

            for image_name in images:
                image_path = os.path.join(class_path, image_name)
                try:

                    image = Image.open(image_path).convert("RGB")


                    target_class_path = os.path.join(target_path, category, class_name)
                    os.makedirs(target_class_path, exist_ok=True)
                    original_image_path = os.path.join(target_class_path, image_name)
                    image.save(original_image_path)


                    for i in range(num_augmented):
                        augmented_image = augmentation_transforms(image)


                        augmented_image_path = os.path.join(
                            target_class_path, f"{image_name.split('.')[0]}_aug{i}.png"
                        )
                        augmented_image_pil = transforms.ToPILImage()(augmented_image)
                        augmented_image_pil.save(augmented_image_path)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")


augment_and_save_images(source_folder, augmented_folder)