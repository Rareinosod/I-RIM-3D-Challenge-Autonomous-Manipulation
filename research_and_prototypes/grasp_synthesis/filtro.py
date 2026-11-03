import cv2
import numpy as np
import os

# Funzione per garantire la creazione della directory di output
def ensure_dir(p):
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# Funzione per pre-processare l'immagine (CLAHE e Gaussian Blur)
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    smooth_image = cv2.GaussianBlur(clahe, (5, 5), 0)
    return smooth_image

# Funzione per applicare GrabCut (funzione di segmentazione basata su semina)
def apply_grabcut(image):
    mask = np.zeros(image.shape[:2], np.uint8)
    rect = (10, 10, image.shape[1]-10, image.shape[0]-10)  # Una zona rettangolare di esempio
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')  # separa oggetto e sfondo
    return mask2

# Funzione per trovare la larghezza dell'impugnatura
def find_grasp_point_from_contours(mask):
    # Trova i contorni dell'oggetto
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None

    # Trova il contorno con la massima larghezza (potrebbe corrispondere all'impugnatura)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Rette tangenti al contorno per trovare la larghezza
    min_rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(min_rect)  # Ottieni i punti del rettangolo
    box = np.int0(box)

    # Calcoliamo il centro del rettangolo come il punto di presa
    center = np.mean(box, axis=0)

    return center

# Funzione per il salvataggio e visualizzazione dei risultati
def save_and_visualize(image, mask, grasp_pos, image_path, output_name="output"):
    # Ottieni la directory dell'immagine di input per il salvataggio dei risultati
    output_dir = os.path.dirname(image_path) + "/output"  # Aggiungi "output" al percorso dell'immagine

    # Assicurati che la cartella di output esista
    ensure_dir(output_dir)  # Aggiungi questa riga per creare la directory di output
    
    result = cv2.bitwise_and(image, image, mask=mask)

    # Disegna il punto di presa sull'immagine
    grasp_pos = tuple(np.round(grasp_pos).astype(int))  # Assicurati che le coordinate siano intere
    cv2.circle(result, grasp_pos, 10, (0, 0, 255), -1)  # Punto di presa in rosso

    # Salva i file di output nella stessa directory dell'immagine
    cv2.imwrite(f"{output_dir}/{output_name}_segmented.png", result)
    cv2.imwrite(f"{output_dir}/{output_name}_mask.png", mask)

    # Scrivi le informazioni sul punto di presa (coordinate)
    with open(f"{output_dir}/{output_name}_coordinates.txt", 'w') as f:
        f.write(f"Grasp Point (pixels): {grasp_pos}\n")

    cv2.imshow("Segmented Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Funzione principale che esegue l'intero flusso
def refine_grasp_on_cropped(image_path, out_prefix="out/a2", visualize=True):
    # Carica l'immagine
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Immagine non trovata: {image_path}")
    
    # Pre-processamento e segmentazione
    smooth_image = preprocess_image(image)

    # Applicazione dei filtri paralleli
    canny_edges = cv2.Canny(smooth_image, 100, 200)  # Canny Edge detection
    _, otsu_thresh = cv2.threshold(smooth_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Otsu thresholding
    grabcut_mask = apply_grabcut(image)  # Applicazione GrabCut

    # Combinazione delle maschere
    combined_mask = cv2.bitwise_or(canny_edges, otsu_thresh)
    combined_mask = cv2.bitwise_or(combined_mask, grabcut_mask)

    # Trova il punto di presa dalla larghezza dell'impugnatura
    grasp_pos = find_grasp_point_from_contours(combined_mask)

    # Salvataggio e visualizzazione dei risultati
    if visualize:
        save_and_visualize(image, combined_mask, grasp_pos, image_path, output_name="grasp_result")

    return grasp_pos

# ESECUZIONE DIRETTA
if __name__ == "__main__":
    IMAGE_PATH = "/Users/alessandro/Desktop/IRIM_Challenge2025/codici python_visual_recognition/box.png"  # Inserisci il percorso dell'immagine
    res = refine_grasp_on_cropped(IMAGE_PATH, out_prefix="grisp/a2", visualize=True)

    print("\n=== Grasping Result ===")
    print(f"Grasp point (pixels): {res}")
    print(f"Salvati: {os.path.dirname(IMAGE_PATH)}/output/grasp_result_segmented.png , grasp_result_mask.png , grasp_result_coordinates.txt")