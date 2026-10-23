import argparse
import os
import glob
import cv2
import numpy as np


def convert_to_transmission(input_dir, output_dir, beta=3.0):
    """Convert DepthAnything output maps to optical transmission maps.

    Args:
        input_dir: Directory containing DepthAnything depth maps (values in [0, 255]).
        output_dir: Output directory for converted transmission maps.
        beta: Attenuation coefficient (default: 3.0).
    """
    os.makedirs(output_dir, exist_ok=True)
    depth_files = glob.glob(os.path.join(input_dir, "*.*"))
    converted_count = 0

    for file_path in depth_files:
        filename = os.path.basename(file_path)

        # Read depth map in grayscale (V in [0, 255])
        V = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if V is None:
            continue

        # Convert to relative distance Z in [0, 1] (brighter pixels represent closer distances)
        Z = 1.0 - (V.astype(np.float32) / 255.0)

        # Transmission formulation: t = exp(-beta * Z)
        t = np.exp(-beta * Z)

        # Quantize to 8-bit image
        t_img = np.clip(t * 255.0, 0, 255).astype(np.uint8)

        out_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.png')
        cv2.imwrite(out_path, t_img)
        converted_count += 1

    print(f"成功转换了 {converted_count} 张传输图到 {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DepthAnything out to Transmission Maps")
    parser.add_argument("--input", required=True, help="Path to folder containing DepthAnything 0-255 images")
    parser.add_argument("--output", required=True, help="Path to save transmission maps (e.g. data/train/transmission)")
    parser.add_argument("--beta", type=float, default=3.0, help="Attenuation coefficient default 3.0")
    args = parser.parse_args()

    convert_to_transmission(args.input, args.output, args.beta)
