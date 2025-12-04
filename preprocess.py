import cv2
import numpy as np
import os
from mtcnn.mtcnn import MTCNN

# --- NEW: Initialize the MTCNN detector ---
# This is a much more accurate, deep-learning-based
# face detector than the old Haar Cascade.
detector = MTCNN()

# Define the final image size for our CNN
OUTPUT_SIZE = 28

# Define input-output folder mapping for all sets
DATA_FOLDERS = [
    ('Dataset/training_data/locksmoking', 'Dataset/training_data/smoking'),
    ('Dataset/training_data/locknotsmoking', 'Dataset/training_data/notsmoking'),
    ('Dataset/validation_data/locksmoking', 'Dataset/validation_data/smoking'),
    ('Dataset/validation_data/locknotsmoking', 'Dataset/validation_data/notsmoking'),
    ('Dataset/testing_data/locksmoking', 'Dataset/testing_data/smoking'),
    ('Dataset/testing_data/locknotsmoking', 'Dataset/testing_data/notsmoking')
]

def format_image(image):
    # --- NEW: MTCNN requires RGB, but OpenCV loads BGR ---
    # We convert to RGB for detection, but we'll crop the original grayscale image.
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run detection
    results = detector.detect_faces(rgb_image)
    
    if not results:
        print("No faces detected")
        return None

    # --- NEW: Filter by confidence instead of just size ---
    # This helps filter out bad detections (hands, hair, etc.)
    best_face = max(results, key=lambda r: r['confidence'])
    
    if best_face['confidence'] < 0.90:
        print(f"Face detection confidence too low ({best_face['confidence']:.2f}). Skipping.")
        return None

    # Get bounding box. MTCNN can return negative coords, so clip to 0.
    x, y, w, h = best_face['box']
    x, y = max(0, x), max(0, y)
    
    print(f"Face detected with {best_face['confidence']:.2f} confidence at: x={x}, y={y}, w={w}, h={h}")
    
    # --- FIX 1: Crop the FACE from the GRAYSCALE image ---
    face_crop = gray_image[y:y+h, x:x+w]
    
    if face_crop.size == 0:
        print("Face crop resulted in empty image. Skipping.")
        return None

    # --- FIX 2: Resize the CROP directly to 28x28. NO BORDER. ---
    try:
        final_image = cv2.resize(face_crop, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)
    except Exception as e:
        print(f"[+] Problem during resize: {e}")
        return None

    # Normalize the image (0.0 to 1.0) for saving
    final_image = final_image.astype(np.float32) / 255.0

    return final_image

def preprocess_images(INPUT_FOLDER, OUTPUT_FOLDER):
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    processed_count = 0
    error_count = 0
    
    for file_name in os.listdir(INPUT_FOLDER):
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue
            
        image_path = os.path.join(INPUT_FOLDER, file_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"[!] Could not read {file_name}")
            error_count += 1
            continue
            
        print(f"--- Processing: {file_name} ---")
        preprocessed_image = format_image(image)
        
        if preprocessed_image is not None:
            output_path = os.path.join(OUTPUT_FOLDER, file_name)
            # Convert back to 0-255 range for saving
            save_image = (preprocessed_image * 255.0).astype(np.uint8)
            cv2.imwrite(output_path, save_image)
            print(f"Processed image saved to {output_path}")
            processed_count += 1
        else:
            print(f"Error processing image {file_name}. Skipping.")
            error_count += 1
    
    print(f"Preprocessing completed for {INPUT_FOLDER}: {processed_count} processed, {error_count} errors")

# Run preprocessing for all folders
for inp, out in DATA_FOLDERS:
    print(f"\n======================================")
    print(f"Processing folder: {inp} -> {out}")
    print(f"======================================")
    preprocess_images(inp, out)

print("\nAll preprocessing finished.")