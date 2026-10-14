"""Read a crop in augmented coordinates without transforming a complete HR image."""

from augmentation import apply_transform


def crop_transformed(image, box, params):
    """Map (top, left, height, width) back to raw pixels, then transform the tile."""
    top, left, height, width = box
    source_width, source_height = image.size
    k = params.rotation_k
    target_width, target_height = ((source_height, source_width) if k % 2
                                   else (source_width, source_height))
    if (height <= 0 or width <= 0 or top < 0 or left < 0
            or top + height > target_height or left + width > target_width):
        raise ValueError("Invalid HR crop")

    # Invert the final counterclockwise rotation using half-open pixel bounds.
    right, bottom = left + width, top + height
    if k == 1:
        left, top, right, bottom = source_width - bottom, left, source_width - top, right
    elif k == 2:
        left, top, right, bottom = (source_width - right, source_height - bottom,
                                   source_width - left, source_height - top)
    elif k == 3:
        left, top, right, bottom = top, source_height - right, bottom, source_height - left

    # The forward order is horizontal flip, vertical flip, then rotation.
    if params.vertical_flip:
        top, bottom = source_height - bottom, source_height - top
    if params.horizontal_flip:
        left, right = source_width - right, source_width - left
    return apply_transform(image.crop((left, top, right, bottom)), params)
