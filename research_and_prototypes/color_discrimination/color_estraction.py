"""
Questo codice classifica immagini in ROSSO, GIALLO o BLU.
Applica CLAHE per stabilizzare la luminosità e converte in HSV.
Crea maschere per ciascun colore con intervalli di Hue e soglie minime su Saturation e Value.
Pulisce le maschere con operazioni morfologiche e seleziona la componente più grande.
Il colore finale è quello con area maggiore; in alternativa usa la tinta media (Hue dominante).
Intervalli scelti: ROSSO [0..10]∪[170..179], GIALLO [20..35], BLU [100..140], con S ≥ 60, V ≥ 50.
Stampa per ogni immagine: "Immagine X rappresenta un oggetto <COLORE>".
"""


import os
import cv2
import numpy as np



IMG_DIR = r"/Users/alessandro/Desktop/IRIM_Challenge2025/codici python_visual_recognition"   
FILENAMES = ["cacciaviteblu.jpeg", "cacciavitegiallo.jpeg", "cacciaviterosso.jpeg"]  


COLOR_RANGES_HSV = {
    # rosso ha due finestre per via del wrap a 0/179
    "rosso": [ (np.array([  0, 60, 50]), np.array([ 10, 255, 255])),
               (np.array([170, 60, 50]), np.array([179, 255, 255])) ],
    "giallo": [ (np.array([ 20, 60, 50]), np.array([ 35, 255, 255])) ],
    "blu":    [ (np.array([100, 60, 50]), np.array([140, 255, 255])) ],
}

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
MIN_MASK_PIXELS = 500   # soglia minima area per considerare valida una maschera


def clahe_on_L(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    Lc = clahe.apply(L)
    labc = cv2.merge([Lc, a, b])
    return cv2.cvtColor(labc, cv2.COLOR_Lab2BGR)

def build_mask_hsv(hsv, ranges):
    """Costruisce una maschera OR di uno o più intervalli HSV (lista di tuple (low,high))."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for low, high in ranges:
        mask |= cv2.inRange(hsv, low, high)
    # pulizia rapida
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=1)
    return mask

def largest_component(mask):
    """Tiene solo la componente connessa più grande (se esiste)."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask, 0
    # stats: [label, x, y, w, h, area] con stats[:, cv2.CC_STAT_AREA]
    areas = stats[1:, cv2.CC_STAT_AREA]  # salta background
    idx = 1 + np.argmax(areas)
    out = np.zeros_like(mask)
    out[labels == idx] = 255
    return out, int(areas.max())

def circular_mean_deg(hue_vals, weights=None):
    """Media circolare su hue (gradi in [0,180))."""
    if len(hue_vals) == 0:
        return None
    ang = np.deg2rad(hue_vals)
    w = np.ones_like(ang) if weights is None else weights.astype(np.float32)
    x = np.sum(w*np.cos(ang))
    y = np.sum(w*np.sin(ang))
    mean = np.arctan2(y, x)
    return (np.rad2deg(mean)) % 180.0

def fallback_hue_vote(hsv, fg_mask):
    """Se i range non catturano, stima colore dominante con istogramma di Hue pesato da Saturation."""
    H = hsv[:,:,0].astype(np.float32)   # [0..179]
    S = hsv[:,:,1].astype(np.float32)
    m = fg_mask.astype(bool)
    if np.count_nonzero(m) < MIN_MASK_PIXELS:
        # usa tutta l'immagine (ultimo fallback)
        m = np.ones_like(H, dtype=bool)

    Hm = H[m]; Sm = S[m]
    # ignora pixel poco saturati
    good = Sm > 25
    if np.count_nonzero(good) < 50:
        good = np.ones_like(Sm, dtype=bool)

    H_sel = Hm[good]
    S_sel = Sm[good]
    if H_sel.size == 0:
        return None

    # media circolare pesata
    hmean = circular_mean_deg(H_sel, weights=S_sel+1)
    if hmean is None:
        return None

    # mappa per vicinanza ai 3 centri (in gradi)
    centers = {"rosso":   [0.0, 179.0],   # gestisci wrap
               "giallo":  [27.5],
               "blu":     [120.0]}
    def hue_dist(a,b):
        d = abs(a-b)
        return min(d, 180-d)

    scores = {}
    for lab, cs in centers.items():
        scores[lab] = min(hue_dist(hmean, c) for c in cs)
    # scegli distanza minima
    label = min(scores, key=scores.get)
    return label

def classify_one_image(img_bgr):
    """Ritorna ('rosso'|'giallo'|'blu', debug_dict)"""
    # 1) normalizza la luminanza (più stabilità colore)
    img_n = clahe_on_L(img_bgr)
    hsv = cv2.cvtColor(img_n, cv2.COLOR_BGR2HSV)

    # 2) prova maschere per ogni colore e prendi la componente più grande
    areas = {}
    masks = {}
    for lab, ranges in COLOR_RANGES_HSV.items():
        mask = build_mask_hsv(hsv, ranges)
        mask_largest, area = largest_component(mask)
        masks[lab] = mask_largest
        areas[lab] = area

    # 3) decisione primaria: colore con area maggiore (se sufficiente)
    best = max(areas, key=areas.get)
    if areas[best] >= MIN_MASK_PIXELS:
        return best, {"areas": areas}

    # 4) fallback: stima hue dominante (pesata da S) nell’area foreground (OR di tutte le maschere)
    fg = np.zeros_like(next(iter(masks.values())))
    for m in masks.values():
        fg |= m
    vote = fallback_hue_vote(hsv, fg)
    if vote is not None:
        return vote, {"areas": areas, "fallback": True}

    # 5) se proprio nulla indeciso 
    return "indeciso", {"areas": areas, "fallback": True, "note": "bassa confidenza"}

if __name__ == "__main__":
    for fname in FILENAMES:
        path = os.path.join(IMG_DIR, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"[ERRORE] Impossibile leggere: {path}")
            continue
        label, _ = classify_one_image(img)
        print(f"Immagine {fname} rappresenta un oggetto {label.upper()}")