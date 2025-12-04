import cv2
import numpy as np
import os
import sys
from mtcnn.mtcnn import MTCNN
import torch
from torchvision import transforms
from PIL import Image

# --- Import Preprocessing ---
try:
    import preprocess_2_new as pp2
except ImportError:
    print("Error: Could not find 'preprocess_2_new.py'.")
    exit()

# --- Config ---
OUTPUT_SIZE = 28
MODEL_LOAD_PATH = "smoking_cnn_model.pth"

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def quantize_8bit_signed(data_float):
    data_scaled = data_float * 127.0
    data_clamped = np.clip(data_scaled, -128.0, 127.0)
    data_int = data_clamped.astype(np.int8)
    return data_int

def main(image_path):
    print(f"--- Processing: {image_path} ---")
    
    if not os.path.exists(image_path):
        print("Error: File not found.")
        return
    
    # 1. Load & Detect
    image = cv2.imread(image_path)
    detector = MTCNN()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    results = detector.detect_faces(rgb_image)
    if not results:
        print("No face detected. Using center crop.")
        face_resized = cv2.resize(gray_image, (28, 28))
    else:
        best_face = max(results, key=lambda r: r['confidence'])
        x, y, w, h = best_face['box']
        x, y = max(0, x), max(0, y)
        face_crop = gray_image[y:y+h, x:x+w]
        face_resized = cv2.resize(face_crop, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)

    # ---- SAVE PREPROCESSED IMAGE ----
    save_path = "preprocessed.png"
    cv2.imwrite(save_path, face_resized)
    print(f"Saved preprocessed 28×28 image as: {save_path}")

    # 2. Python Prediction
    print("Checking Python Prediction...")
    face_pil = Image.fromarray(face_resized)
    tensor_image = transform(face_pil)
    
    model = pp2.CNNModel()
    model.load_state_dict(torch.load(MODEL_LOAD_PATH, weights_only=True))
    model.eval()
    
    with torch.no_grad():
        output = model(tensor_image.unsqueeze(0))
        _, predicted_idx = torch.max(output.data, 1)
        class_name = pp2.train_dataset.classes[predicted_idx[0]]
    
    print(f"PYTHON PREDICTION: {class_name}")

    # 3. Generate Verilog Code
    print("\n--- COPY THE CODE BELOW INTO conv1_buf.v ---")
    
    flattened = tensor_image.numpy().flatten()
    quantized = quantize_8bit_signed(flattened)
    data_uint = quantized.astype(np.uint8)
    
    for i, val in enumerate(data_uint):
 #       print(f"assign data_in[{i}] = 8'h{val:02x};")
        # NEW FORMAT (Correct for cnn_top)
        print(f"img_mem[{i}] = 8'h{val:02x};")
        
    print("--- END OF CODE ---")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python image_to_hex_verilog.py \"path/to/image.jpg\"")
    else:
        main(sys.argv[1])
