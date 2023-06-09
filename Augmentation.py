from PIL import Image
import os

# Directory paths
original_dir = 'Severity/healthy/'
augmented_dir = 'spiralimages/path_to_save_augmented_images_HC/'

# Augmentation parameters
rotation_angles = [90, 180, 270]  # Rotation angles in degrees
flip_modes = ['horizontal', 'vertical']  # Flip modes
resize_dims = [(300, 300), (500, 500)]  # Resizing dimensions

# Create the augmented directory if it doesn't exist
if not os.path.exists(augmented_dir):
    os.makedirs(augmented_dir)

# Iterate through each image in the original directory
for filename in os.listdir(original_dir):
    image_path = os.path.join(original_dir, filename)
    
    # Open the image
    with Image.open(image_path) as img:
        # Apply rotation augmentation
        for angle in rotation_angles:
            rotated_img = img.rotate(angle)
            rotated_img.save(os.path.join(augmented_dir, f'rotated_{angle}_{filename}'))

        # Apply flipping augmentation
        for mode in flip_modes:
            flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT) if mode == 'horizontal' else img.transpose(Image.FLIP_TOP_BOTTOM)
            flipped_img.save(os.path.join(augmented_dir, f'flipped_{mode}_{filename}'))

        # Apply resizing augmentation
        for dims in resize_dims:
            resized_img = img.resize(dims)
            resized_img.save(os.path.join(augmented_dir, f'resized_{dims[0]}_{dims[1]}_{filename}'))
