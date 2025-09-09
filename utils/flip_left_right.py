import os
import argparse
from PIL import Image

def flip_images(input_root, output_root):
    # Walk through all subdirectories and files
    for dirpath, _, filenames in os.walk(input_root):
        # Compute corresponding output path
        relative_path = os.path.relpath(dirpath, input_root)
        out_dir = os.path.join(output_root, relative_path)
        os.makedirs(out_dir, exist_ok=True)

        for filename in filenames:
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")):
                in_path = os.path.join(dirpath, filename)
                out_path = os.path.join(out_dir, filename)

                # Open, flip, and save
                img = Image.open(in_path)
                mirrored = img.transpose(Image.FLIP_LEFT_RIGHT)
                mirrored.save(out_path)

    print("✅ Done! Flipped images saved under", output_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flip images left-right recursively.")
    parser.add_argument('--input_root', type=str, required=True,
                        help='Path to the root folder of input images (e.g., comma/rgb-images)')
    parser.add_argument('--output_root', type=str, required=True,
                        help='Path to save the flipped images (e.g., comma_flipped/rgb-images)')
    args = parser.parse_args()

    flip_images(args.input_root, args.output_root)