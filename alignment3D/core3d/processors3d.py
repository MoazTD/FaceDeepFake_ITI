import cv2
import numpy as np
from PIL import Image

def manual_blend(original: np.ndarray, swapped: np.ndarray, bbox: np.ndarray, alpha: float, softness: float, custom_mask: np.ndarray = None) -> np.ndarray:
    """
    Blends the 'swapped' image into the 'original' image.
    Uses a custom mask if provided, otherwise creates a default elliptical mask.
    """
    x1, y1, x2, y2 = bbox.astype(int)
    h, w = y2 - y1, x2 - x1

    if custom_mask is not None and w > 0 and h > 0:
        # The custom_mask is a "face-relative" template.
        # We resize it to fit the bounding box of the face in the current frame.
        mask_template_img = Image.fromarray(custom_mask)
        resized_mask_img = mask_template_img.resize((w, h), Image.Resampling.NEAREST)
        
        # Create a full-sized mask and place the resized face mask onto it
        mask = np.zeros(original.shape[:2], dtype=np.uint8)
        mask[y1:y2, x1:x2] = np.array(resized_mask_img)
    else:
        # Fallback to a default white elliptical mask inside the bounding box
        mask = np.zeros(original.shape[:2], dtype=np.uint8)
        center = (x1 + w // 2, y1 + h // 2)
        axes = (w // 2, h // 2)
        cv2.ellipse(mask, center, axes, angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)

    # Blur the mask to create soft edges, kernel size must be odd
    k = max(1, int(softness) // 2 * 2 + 1)
    blurred_mask = cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32) / 255.0
    
    # Adjust mask by alpha and ensure it's in the right shape for broadcasting
    final_mask = np.clip(blurred_mask * alpha, 0, 1)[..., None]

    # Blend the images using the soft mask
    orig_f = original.astype(np.float32)
    swap_f = swapped.astype(np.float32)
    out = orig_f * (1 - final_mask) + swap_f * final_mask
    
    return out.astype(np.uint8)


def _match_channel_stats(source_channel: np.ndarray, target_channel: np.ndarray) -> np.ndarray:
    """Helper function to match the mean and std dev of a target channel."""
    source_channel = source_channel.astype(np.float32)
    
    mean_src, std_src = cv2.meanStdDev(source_channel)
    mean_tgt, std_tgt = cv2.meanStdDev(target_channel)

    if std_src[0,0] < 1: 
        std_src[0,0] = 1

    corrected_channel = source_channel - mean_src
    corrected_channel = (std_tgt[0,0] / std_src[0,0]) * corrected_channel
    corrected_channel = corrected_channel + mean_tgt
    corrected_channel = np.clip(corrected_channel, 0, 255)
    
    return corrected_channel.astype(np.uint8)

def apply_color_correction(original_img: np.ndarray, swapped_face_img: np.ndarray, target_face_bbox: np.ndarray) -> np.ndarray:
    """
    Applies color correction to a swapped face by matching the color distribution
    of the original face in the target image.
    """
    x1, y1, x2, y2 = target_face_bbox.astype(int)
    original_face = original_img[y1:y2, x1:x2]
    swapped_face = swapped_face_img[y1:y2, x1:x2]

    original_face_lab = cv2.cvtColor(original_face, cv2.COLOR_BGR2LAB)
    swapped_face_lab = cv2.cvtColor(swapped_face, cv2.COLOR_BGR2LAB)

    l_orig, a_orig, b_orig = cv2.split(original_face_lab)
    l_swap, a_swap, b_swap = cv2.split(swapped_face_lab)

    a_corrected = _match_channel_stats(a_swap, a_orig)
    b_corrected = _match_channel_stats(b_swap, b_orig)

    corrected_lab = cv2.merge([l_swap, a_corrected, b_corrected])
    corrected_face_bgr = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
    
    result_img = swapped_face_img.copy()
    result_img[y1:y2, x1:x2] = corrected_face_bgr
    return result_img
