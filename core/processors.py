try:
    from core.color_blender import apply_color_correction, manual_blend
except ImportError:
    
    def apply_color_correction(original_img, swapped_face_img, target_face_bbox):
        return swapped_face_img
    
    def manual_blend(original, swapped, bbox, alpha, softness):
        x1, y1, x2, y2 = bbox.astype(int)
        result = original.copy()
        result[y1:y2, x1:x2] = swapped[y1:y2, x1:x2]
        return result


__all__ = ['apply_color_correction', 'manual_blend']


def apply_face_enhancement(image, enhancement_type='quality', strength=50):
    import cv2
    import numpy as np
    
    if enhancement_type == 'sharpness':
        
        gaussian = cv2.GaussianBlur(image, (0, 0), strength / 10)
        enhanced = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        
    elif enhancement_type == 'smoothing':
        
        enhanced = cv2.bilateralFilter(image, int(strength / 5), strength, strength)
        
    elif enhancement_type == 'color':
        
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        
        a = cv2.add(a, int(strength / 2))
        b = cv2.add(b, int(strength / 2))
        
        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
    else:  
        
        
        denoised = cv2.fastNlMeansDenoisingColored(image, None, strength / 10, strength / 10, 7, 21)
        
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(denoised, -1, kernel)
    
    return enhanced


def create_face_mask(face_bbox, image_shape, expand_ratio=1.2):
    import cv2
    import numpy as np
    
    x1, y1, x2, y2 = face_bbox.astype(int)
    h, w = image_shape[:2]
    
    
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    w_half = int((x2 - x1) * expand_ratio / 2)
    h_half = int((y2 - y1) * expand_ratio / 2)
    
    x1 = max(0, cx - w_half)
    x2 = min(w, cx + w_half)
    y1 = max(0, cy - h_half)
    y2 = min(h, cy + h_half)
    
    
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    axes = ((x2 - x1) // 2, (y2 - y1) // 2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    
    
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    
    return mask
