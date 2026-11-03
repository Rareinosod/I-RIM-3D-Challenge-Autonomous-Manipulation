import cv2
import numpy as np
import logging
import os
import glob # Needed to find files in folders

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- MODIFY THESE TWO FOLDER PATHS ---
PATH_FOLDER_RED = "/Users/alessandro/Desktop/IRIM_Challenge2025/codici python_visual_recognition/TATA/rosso" # Folder with red/yellow crops
PATH_FOLDER_BLACK = "/Users/alessandro/Desktop/IRIM_Challenge2025/codici python_visual_recognition/TATA/nero" # Folder with black/blue crops
# --- -------------------------------- ---

# Name of the file where the average templates will be saved
OUTPUT_FILE = "/Users/alessandro/Desktop/IRIM_Challenge2025/codici python_visual_recognition/TATA/impronte_medie_salvate.py"

def calculate_average_hist_from_folder(folder_path):
    """Calculates the average histogram from all images in a folder."""
    histograms = []
    image_patterns = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_files = []
    
    if not os.path.isdir(folder_path):
        logging.error(f"Folder not found: {folder_path}")
        return None
        
    # Find all image files in the folder
    for pattern in image_patterns:
        image_files.extend(glob.glob(os.path.join(folder_path, pattern)))

    if not image_files:
        logging.error(f"No image files found in folder: {folder_path}")
        return None

    logging.info(f"Found {len(image_files)} images in {os.path.basename(folder_path)} folder.")

    # Calculate histogram for each image
    for image_path in image_files:
        img = cv2.imread(image_path)
        if img is None:
            logging.warning(f"Could not read image: {image_path}")
            continue
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Filter: Keep only sufficiently saturated and bright pixels
        # Using the less aggressive filters that seemed promising
        s_min = 45 
        v_min = 25 
        mask = (s > s_min) & (v > v_min)
        
        # Check if there are any valid pixels after filtering
        if np.sum(mask) == 0:
            logging.warning(f"No valid colored pixels found in {os.path.basename(image_path)}. Skipping.")
            continue
            
        # Calculate HUE histogram (180 bins)
        hist = cv2.calcHist([hsv], [0], mask.astype(np.uint8), [180], [0, 180])
        
        # Check if histogram calculation was successful and has the correct shape
        if hist is not None and hist.shape == (180, 1):
             cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX) # Normalize immediately
             histograms.append(hist)
        else:
            logging.warning(f"Invalid histogram calculated for {os.path.basename(image_path)}. Skipping.")

    # Check if we got any valid histograms
    if not histograms:
        logging.error(f"No valid histograms could be calculated for folder {folder_path}.")
        return None

    # Calculate the average of all collected histograms
    logging.info(f"Calculating average histogram from {len(histograms)} valid images.")
    average_hist = np.mean(histograms, axis=0).astype(np.float32)
    
    # Re-normalize the final average histogram (important!)
    cv2.normalize(average_hist, average_hist, 0, 1, cv2.NORM_MINMAX)
    
    return average_hist

# --- CALCULATE AVERAGE HISTOGRAMS ---
logging.info(f"Calculating average template for (RED) from folder: {PATH_FOLDER_RED}")
hist_red_avg = calculate_average_hist_from_folder(PATH_FOLDER_RED)

logging.info(f"Calculating average template for (BLACK) from folder: {PATH_FOLDER_BLACK}")
hist_black_avg = calculate_average_hist_from_folder(PATH_FOLDER_BLACK)

# --- SAVE TO FILE ---
if hist_red_avg is not None and hist_black_avg is not None:
    # Use repr() for a safe string representation that includes dtype
    red_str = repr(hist_red_avg)
    black_str = repr(hist_black_avg)

    # Write the numpy array definitions to the output file
    try:
        with open(OUTPUT_FILE, "w") as f:
            f.write("# File generated automatically by genera_impronte.py (average version)\n")
            f.write("# Contains the AVERAGE histogram templates for the screwdrivers\n\n")
            f.write("import numpy as np # Make sure numpy is imported in your main script\n\n")
            # Write the arrays using their repr() string
            f.write(f"self.hist_red = np.{red_str}\n\n") 
            f.write(f"self.hist_black = np.{black_str}\n")
        
        logging.info(f"\n--- AVERAGE TEMPLATES SAVED SUCCESSFULLY TO: '{OUTPUT_FILE}' ---")
        logging.info(f"--- Open '{OUTPUT_FILE}', copy the two 'self.hist_...' lines ---")
        logging.info(f"--- and paste them into your main script ('YOLO.py', inside __init__) ---")

    except Exception as e:
        logging.error(f"Error writing to output file '{OUTPUT_FILE}': {e}")

else:
    logging.error("Failed to calculate one or both average histograms. Check previous messages and folders.")