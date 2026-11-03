'''
INPUT: questo codice usa OUTPUT del codice: tool_table_finale in particoalre prende la fotografia del tool desiderato 
e tramite delle teniche di preprocessing e filtraggio risce a ricavare una maschera del tool.
a seguito il codice implmneta un logica per creare due assi  ( il PCA e l'asse ortogonale ) e da esso ricava la ccordinata del punto di grip migliore

OUTPUT : abbiamo le coordinate del punto piu comodo per afferare il tool desiderato 
'''

import cv2
import numpy as np
import math
import os

# ====== ========= ======
IMAGE_PATH = "/Users/alessandro/Desktop/IRIM_Challenge2025/codici python_visual_recognition/box.png"  
OUT_PREFIX = "immagine_filtrata/IMG_TOOL"  

# Utility per la gestione delle directory
def ensure_dir(p):
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# Funzione per ottenere la maschera della componente connessa più grande
def largest_component_mask(binary):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + np.argmax(areas)
    mask = (labels == best).astype(np.uint8) * 255
    return mask

# Calcola gli assi principali usando PCA
def pca_axis_from_mask(mask):
    ys, xs = np.where(mask > 0)
    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    if len(pts) < 10:
        return None, None, None
    mean, eigenvectors = cv2.PCACompute(pts, mean=None)
    center = (float(mean[0,0]), float(mean[0,1]))
    u = eigenvectors[0]
    u = u / (np.linalg.norm(u) + 1e-9)  # Normalizzazione
    v = np.array([-u[1], u[0]], dtype=np.float32)  # Direzione ortogonale
    return center, u.astype(np.float32), v.astype(np.float32)

# Profilazione della larghezza lungo l'asse principale dell'oggetto
def profile_width_along_axis(mask, center, u, v, nbins=60):
    ys, xs = np.where(mask > 0)
    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    rel = pts - np.array(center, dtype=np.float32)[None, :]
    s = rel @ u
    t = rel @ v
    s_min, s_max = np.min(s), np.max(s)
    bins = np.linspace(s_min, s_max, nbins+1)
    bin_idx = np.clip(np.digitize(s, bins) - 1, 0, nbins-1)

    t_min = np.full(nbins, np.inf, dtype=np.float32)
    t_max = np.full(nbins, -np.inf, dtype=np.float32)
    counts = np.zeros(nbins, dtype=np.int32)
    for i in range(len(s)):
        b = bin_idx[i]
        tt = t[i]
        if tt < t_min[b]: t_min[b] = tt
        if tt > t_max[b]: t_max[b] = tt
        counts[b] += 1
    width = np.zeros(nbins, dtype=np.float32)
    valid = counts > 0
    width[valid] = (t_max[valid] - t_min[valid])
    k = 5
    if nbins >= k:
        kernel = np.ones(k, dtype=np.float32)/k
        width = np.convolve(width, kernel, mode='same')
    s_centers = 0.5 * (bins[:-1] + bins[1:])
    return {
        "s_centers": s_centers, "width": width,
        "t_min": t_min, "t_max": t_max,
        "valid": valid, "s_min": s_min, "s_max": s_max
    }

def pick_grasp_from_profile(profile, exclude_ends_frac=0.10):
    """
    Seleziona il miglior punto di presa escludendo le estremità dell'oggetto
    in base alla larghezza lungo l'asse principale. Puoi escludere un
    determinato frazione  all'inizio e alla fine dell'oggetto.
    
    exclude_ends_frac: frazione da escludere dalle estremità dell'oggetto.
    """
    s = profile["s_centers"]   # I punti lungo l'asse principale (s)
    w = profile["width"]       # Larghezza in ogni bin
    valid = profile["valid"]   # Punti validi
    s_min, s_max = profile["s_min"], profile["s_max"]  # Limiti dell'oggetto lungo l'asse principale
    span = s_max - s_min

    # Calcola le soglie per escludere le estremità (10% inizio e 10% fine)
    low = s_min + exclude_ends_frac * span
    high = s_max - exclude_ends_frac * span

    # Seleziona solo le aree valide che non siano nelle estremità
    mask_ok = (s >= low) & (s <= high) & valid & (w > 0)

    # Se non ci sono aree valide, prova ad usare qualsiasi punto valido (senza esclusione)
    if not np.any(mask_ok):
        mask_ok = valid & (w > 0)
        if not np.any(mask_ok):
            return None

    # Trova il bin con la larghezza massima nella regione selezionata
    idx = np.argmax(w[mask_ok])  # Troviamo l'indice del punto con la larghezza massima
    abs_indices = np.where(mask_ok)[0]  # Troviamo gli indici assoluti dei punti validi
    return int(abs_indices[idx])  # Restituiamo l'indice del miglior punto di presa

# Funzione per fare il backprojection dei punti di presa
def backproject(center, u, v, s, t):
    pt = np.array(center) + s * u + t * v
    return float(pt[0]), float(pt[1])

# ====== FILTRI AGGIUNTIVI ======
def apply_gaussian_process_filter(image):
    # Filtro Gaussian per la segmentazione
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    return blur



def apply_bilateral_filter(image):
    # Filtro bilaterale per ridurre il rumore e mantenere i bordi
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    return bilateral



# ====== main ======
def refine_grasp_on_cropped(image_path, out_prefix="out/a2", visualize=True):
    ensure_dir(out_prefix + "_dummy.png")
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Immagine non trovata: {image_path}")
    h, w = bgr.shape[:2]

    # Pre-processamento dell'immagine 

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    kernel = np.ones((2, 2), np.uint8)
    hsv = cv2.dilate(hsv, kernel, iterations=3)

    gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)


    kernel = np.ones((2, 2), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=3)

    clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(5,5)).apply(gray)


    kernel = np.ones((2, 2), np.uint8)
    clahe = cv2.dilate(clahe, kernel, iterations=1)



    #filtri

    blur=apply_gaussian_process_filter(clahe)
    bilater=apply_bilateral_filter(blur)


    
    
    
    # Soglia binaria usando Otsu
    _, th = cv2.threshold(bilater, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    kernel = np.ones((4, 4), np.uint8)
    th = cv2.dilate(th, kernel, iterations=3)
    


   
    def best_mask(candidate):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        m = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
        m = largest_component_mask(m)
        return m

    
    mask = best_mask(th)
    


    kernel = np.ones((1, 1), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    

    # Calcola l'orientamento e il centro dell'oggetto
    center, u, v = pca_axis_from_mask(mask)
    if center is None:
        raise RuntimeError("Maschera troppo piccola.")
    
    # Profilazione della larghezza dell'oggetto lungo l'asse principale
    prof = profile_width_along_axis(mask, center, u, v, nbins=60)
    
    # Seleziona il punto di presa dal profilo
    idx = pick_grasp_from_profile(prof, exclude_ends_frac=0.10)
    if idx is None:
        raise RuntimeError("Nessun bin valido trovato.")
    
    s0 = prof["s_centers"][idx]
    t_mid = 0.5 * (prof["t_min"][idx] + prof["t_max"][idx])
    gx, gy = backproject(center, u, v, s0, t_mid)
    
    # Calcola l'orientamento
    theta = math.degrees(math.atan2(u[1], u[0]))

    # Salvataggio delle coordinate e orientamento in un file di testo
    with open(out_prefix + "_grasp_data.txt", "w") as file:
        file.write(f"Grasp Point: ({gx}, {gy})\n")
        file.write(f"Orientation (theta): {theta:.1f} degrees\n")
        file.write(f"Image Size: ({w}, {h})\n")

    print(f"Dati di presa salvati in {out_prefix}_grasp_data.txt")

    # Visualizzazione dei risultati
    if visualize:
        vis = bgr.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0,255,0), 2)
        length = max(w, h)
        p1 = (int(center[0] - u[0]*length), int(center[1] - u[1]*length))
        p2 = (int(center[0] + u[0]*length), int(center[1] + u[1]*length))
        cv2.line(vis, p1, p2, (255,200,0), 2)
        t1, t2 = prof["t_min"][idx], prof["t_max"][idx]
        a1 = backproject(center, u, v, s0, t1)
        a2 = backproject(center, u, v, s0, t2)
        cv2.line(vis, (int(a1[0]), int(a1[1])), (int(a2[0]), int(a2[1])), (255,0,255), 3)
        cv2.circle(vis, (int(round(gx)), int(round(gy))), 6, (0,0,255), -1)
        
        ensure_dir(out_prefix + "_mask.png")
       

        cv2.imwrite(out_prefix + "_debug.png", vis)
        
        
        
        


    return {
        "grasp_px": (gx, gy),
        "angle_deg": theta,
        "image_size": (w, h)
    }

# ====== ESECUZIONE DIRETTA ======
if __name__ == "__main__":
    res = refine_grasp_on_cropped(IMAGE_PATH, OUT_PREFIX, visualize=True)
    print("\n===  RESULT ===")
    print(f"Grasp point (pixels): {res['grasp_px']}")
    print(f"Angle (deg): {res['angle_deg']:.1f}")
    print(f"Image size (w,h): {res['image_size']}")
    print(f"Salvati: {OUT_PREFIX}_mask.png , {OUT_PREFIX}_debug.png , {OUT_PREFIX}_grasp_data.txt")