
import onnxruntime as ort
import numpy as np
import cv2

def preprocess(img, size):
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5  
    img = img.transpose(2, 0, 1)  
    img = np.expand_dims(img, axis=0)
    return img

def postprocess(output, size):
    img = np.clip(output, -1, 1)
    img = (img + 1) / 2  
    img = img.transpose(1, 2, 0)  
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, size)
    return img

def apply_ageing(img: np.ndarray, target_age: int):
    
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    session = ort.InferenceSession(r"Models\styleganex_age.onnx", providers=providers)

    
    print("Model input names and shapes:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.shape}")

    
    img_bg = preprocess(img, (384, 384))
    img_tgt = preprocess(img, (256, 256))

    
    direction = np.interp(target_age, [-100, 100], [2.5, -2.5]).astype(np.float32)
    direction = np.array(direction, dtype=np.float32)

    
    inputs = {
        "target_with_background": img_bg,
        "target": img_tgt,
        "direction": direction
    }

    output = session.run(None, inputs)[0][0]  
    output_img = postprocess(output, (img.shape[1], img.shape[0]))
    return output_img
if __name__ == "__main__":
    import os
    img_path = input("Enter path to your image: ").strip()
    if not os.path.isfile(img_path):
        print("Image file not found!")
        exit(1)

    try:
        target_age = int(input("Enter target age (e.g., 60): ").strip())
    except ValueError:
        print("Invalid age input.")
        exit(1)

    img = cv2.imread(img_path)
    if img is None:
        print("Failed to load image!")
        exit(1)

    aged_img = apply_ageing(img, target_age)
    out_path = f"aged_{target_age}_" + os.path.basename(img_path)
    cv2.imwrite(out_path, aged_img)
    print(f"Aged image saved as {out_path}")
