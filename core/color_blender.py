import cv2
import numpy as np


def manual_blend(original: np.ndarray, swapped: np.ndarray, bbox: np.ndarray, alpha: float, softness: float, custom_mask: np.ndarray = None) -> np.ndarray:

    x1, y1, x2, y2 = bbox.astype(int)
    
    img_h, img_w = original.shape[:2]
    x1 = max(0, min(x1, img_w))
    y1 = max(0, min(y1, img_h))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))
    
    if x1 >= x2:
        x1, x2 = max(0, x2-1), min(img_w, x1+1)
    if y1 >= y2:
        y1, y2 = max(0, y2-1), min(img_h, y1+1)
    
    h, w = y2 - y1, x2 - x1
    
    if w <= 0 or h <= 0:
        print(f"Warning: Invalid face region dimensions: w={w}, h={h}, bbox={bbox}")
        return original
    
    try:
        if custom_mask is not None and custom_mask.size > 0:

            mask = cv2.resize(custom_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)
            center = (w // 2, h // 2)
            axes = (w // 2, h // 2)
            cv2.ellipse(mask, center, axes, angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)
        
        if softness > 0:
            kernel_size = max(1, int(softness / 3) * 2 + 1)
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        
        mask_float = mask.astype(np.float32) / 255.0
        mask_float = np.clip(mask_float * alpha, 0, 1)
        
        orig_face = original[y1:y2, x1:x2].astype(np.float32)
        swap_face = swapped[y1:y2, x1:x2].astype(np.float32)
        
        if orig_face.shape[:2] != mask_float.shape[:2] or swap_face.shape[:2] != mask_float.shape[:2]:
            print(f"Warning: Shape mismatch - orig_face: {orig_face.shape}, swap_face: {swap_face.shape}, mask: {mask_float.shape}")
            return original
        
        if len(orig_face.shape) == 3 and len(mask_float.shape) == 2:
            mask_float = mask_float[..., None]
        
        if orig_face.shape != swap_face.shape:
            print(f"Warning: Face regions have different shapes - orig: {orig_face.shape}, swap: {swap_face.shape}")
            return original
        
        blended_face = orig_face * (1 - mask_float) + swap_face * mask_float
        
        result = original.copy()
        result[y1:y2, x1:x2] = blended_face.astype(np.uint8)
        
        return result
        
    except Exception as e:
        print(f"Error in manual_blend: {e}")
        print(f"Debug info - bbox: {bbox}, h: {h}, w: {w}")
        print(f"Original shape: {original.shape}, Swapped shape: {swapped.shape}")
        if 'mask' in locals():
            print(f"Mask shape: {mask.shape}")
        return original

def apply_face_specific_blending(original: np.ndarray, swapped: np.ndarray, bbox: np.ndarray, 
                                blend_strength: float, edge_softness: float, quality: float = 1.0) -> np.ndarray:

    alpha = blend_strength / 100.0
    strength = quality / 100.0
    
    result = manual_blend(original, swapped, bbox, alpha, edge_softness, strength)
    
    if quality < 80 and edge_softness > 0:
        kernel_size = max(1, int((100 - quality) / 20) * 2 + 1)
        result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0.5)
    
    return result

def _match_channel_stats(source_channel: np.ndarray, target_channel: np.ndarray) -> np.ndarray:
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

    try:
        x1, y1, x2, y2 = target_face_bbox.astype(int)
        
        img_h, img_w = original_img.shape[:2]
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))
        
        if x1 >= x2 or y1 >= y2:
            print("Warning: Invalid bbox for color correction")
            return swapped_face_img
        
        original_face = original_img[y1:y2, x1:x2]
        swapped_face = swapped_face_img[y1:y2, x1:x2]
        
        if original_face.size == 0 or swapped_face.size == 0:
            print("Warning: Empty face region for color correction")
            return swapped_face_img

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
        
    except Exception as e:
        print(f"Error in apply_color_correction: {e}")
        return swapped_face_img