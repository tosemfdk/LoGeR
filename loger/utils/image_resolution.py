import math
import os
from typing import List, Optional, Tuple

from PIL import Image


def resolve_target_image_size(
    image_paths: List[str],
    pixel_limit: int = 255000,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    target_short_side: Optional[int] = None,
) -> Tuple[int, int]:
    if target_width is not None and target_height is not None:
        return target_width, target_height

    first_valid_path = next((img_path for img_path in image_paths if os.path.isfile(img_path)), None)
    if first_valid_path is None:
        raise FileNotFoundError("Could not determine target size because no valid image paths were found.")

    with Image.open(first_valid_path) as first_img:
        first_img = first_img.convert("RGB")
        width_orig, height_orig = first_img.size

    if target_short_side is not None:
        if target_short_side <= 0:
            raise ValueError("Target shorter side must be a positive integer.")
        short_side_orig = min(width_orig, height_orig)
        if short_side_orig <= 0:
            raise ValueError("Input images must have positive width and height.")
        scale = target_short_side / short_side_orig
        width_target, height_target = width_orig * scale, height_orig * scale
        return max(1, round(width_target / 14)) * 14, max(1, round(height_target / 14)) * 14

    scale = math.sqrt(pixel_limit / (width_orig * height_orig)) if width_orig * height_orig > 0 else 1
    width_target, height_target = width_orig * scale, height_orig * scale
    width_steps, height_steps = round(width_target / 14), round(height_target / 14)
    while (width_steps * 14) * (height_steps * 14) > pixel_limit:
        if width_steps / height_steps > width_target / height_target:
            width_steps -= 1
        else:
            height_steps -= 1
    return max(1, width_steps) * 14, max(1, height_steps) * 14


def normalize_target_resolution_arg(target_resolution: Optional[List[int]]) -> Optional[List[int]]:
    if target_resolution is None:
        return None
    if len(target_resolution) not in {1, 2}:
        raise ValueError("--resolution expects one integer (shorter side) or two integers (width height).")
    if any(value <= 0 for value in target_resolution):
        raise ValueError("--resolution values must be positive integers.")
    return target_resolution
