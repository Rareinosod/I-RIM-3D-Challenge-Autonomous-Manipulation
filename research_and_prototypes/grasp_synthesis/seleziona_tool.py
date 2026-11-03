import time
import cv2
import numpy as np
from ultralytics import YOLO

# =============== Config (tarabili) ===============
WEIGHTS_PATH  = "best.pt"
DEVICE        = "cpu"          # "cpu" oppure 0 per GPU
WANTED        = {"hammer", "screwdriver"}  # set() per tutte le classi

# Soglie YOLO (per frame)
CONF_THRES    = 0.60
IOU_THRES_NMS = 0.40

# Filtri area% (camera mobile → larghe)
AREA_MIN_PCT  = 0.003   # 0.3% del frame
AREA_MAX_PCT  = 0.60    # 50% del frame

# Tracking & stabilità
ASSOC_IOU_MIN = 0.50
K_IN          = 3        # conferma dopo >=3 hit
K_OUT         = 8        # disattiva dopo 8 miss
EMA_ALPHA     = 0.30     # smoothing confidenza
SMOOTH_BETA   = 0.60     # smoothing bbox

# Soglia conf media per conferma per classe
CLASS_CONF_MIN = {
    "hammer": 0.62,
    "screwdriver": 0.65,
}
DEFAULT_CONF_MEAN_MIN = 0.62

# “Inventario stabile” (nessun cambiamento per N frame)
STABLE_TICKS   = 5

# Target lock (ri-aggancio se cambia ID)
LOCK_REACQUIRE_MAX_MISS = 30      # quanti frame provare a ri-agganciare
LOCK_REACQUIRE_IOU_MIN  = 0.30    # IoU minima con l’ultima bbox selezionata

# GUI
WINDOW_NAME    = "YOLO — Tools (dynamic camera)"
FONT           = cv2.FONT_HERSHEY_SIMPLEX

# =================================================

def iou_xyxy(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    areaB = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union = areaA + areaB - inter + 1e-9
    return inter / union

def bbox_area_pct(b, W, H):
    return max(0, (b[2]-b[0])) * max(0, (b[3]-b[1])) / float(W*H + 1e-9)

def center_of(b):
    return (0.5*(b[0]+b[2]), 0.5*(b[1]+b[3]))

class Track:
    _next_id = 1
    def __init__(self, cls_name, box, conf):
        self.id = Track._next_id; Track._next_id += 1
        self.cls = cls_name
        self.box = list(map(float, box))          # [x1,y1,x2,y2]
        self.conf_ema = float(conf)
        self.hits = 1
        self.misses = 0
        self.age = 1
        self.confirmed = False
        self.label = None  # es. "hammer_1"

    def update(self, box, conf):
        self.box[0] = SMOOTH_BETA*box[0] + (1-SMOOTH_BETA)*self.box[0]
        self.box[1] = SMOOTH_BETA*box[1] + (1-SMOOTH_BETA)*self.box[1]
        self.box[2] = SMOOTH_BETA*box[2] + (1-SMOOTH_BETA)*self.box[2]
        self.box[3] = SMOOTH_BETA*box[3] + (1-SMOOTH_BETA)*self.box[3]
        self.conf_ema = EMA_ALPHA*conf + (1-EMA_ALPHA)*self.conf_ema
        self.hits += 1
        self.misses = 0
        self.age += 1
        thr = CLASS_CONF_MIN.get(self.cls, DEFAULT_CONF_MEAN_MIN)
        if (not self.confirmed) and self.hits >= K_IN and self.conf_ema >= thr:
            self.confirmed = True

    def mark_miss(self):
        self.misses += 1
        self.age += 1

    def xyxy_int(self):
        return [int(round(x)) for x in self.box]

def associate_tracks(tracks, dets, iou_min):
    if not tracks or not dets:
        return [], list(range(len(tracks))), list(range(len(dets)))
    used_tracks, used_dets = set(), set()
    pairs = []
    while True:
        best = (-1, -1, -1.0)
        for ti, tr in enumerate(tracks):
            if ti in used_tracks: continue
            for dj, (nm, bx, cf) in enumerate(dets):
                if dj in used_dets: continue
                if nm != tr.cls: continue
                iou = iou_xyxy(tr.box, bx)
                if iou > best[2]:
                    best = (ti, dj, iou)
        ti, dj, best_iou = best
        if best_iou < iou_min or ti < 0:
            break
        used_tracks.add(ti); used_dets.add(dj)
        pairs.append((ti, dj))
    unmatched_tracks = [i for i in range(len(tracks)) if i not in used_tracks]
    unmatched_dets   = [j for j in range(len(dets)) if j not in used_dets]
    return pairs, unmatched_tracks, unmatched_dets

# ------------------ Modello ------------------
model = YOLO(WEIGHTS_PATH)
model.overrides["conf"] = CONF_THRES
model.overrides["iou"]  = IOU_THRES_NMS

# ------------------ Webcam ------------------
cap = cv2.VideoCapture(0)  # 0 = webcam di default
if not cap.isOpened():
    raise RuntimeError("Webcam non disponibile")
time.sleep(0.2)

tracks = []
palette_confirmed = (0, 200, 0)   # verde
palette_pending   = (0, 200, 200) # giallo
palette_selected  = (0, 0, 255)   # rosso

# gestione etichette numerate
class_counters = {}         # es. {'hammer': 1, 'screwdriver': 2}
labels_correnti = []        # vettore con i label attuali (solo confermati)
label_to_track  = {}        # mappa label -> Track (confermato)

# stabilità inventario
stable_counter = 0
last_label_set = set()

# selezione / lock
selection_mode   = False
selected_label   = None
selected_class   = None
selected_track   = None
selected_lastbox = None
lock_miss_count  = 0

print("Premi 'q' o ESC per uscire. Premi 'r' per resettare la selezione.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # ---- FLIP orizzontale ----
    frame = cv2.flip(frame, 1)
    H, W = frame.shape[:2]

    # ---- YOLO inference ----
    r = model.predict(source=frame, device=DEVICE, verbose=False)[0]
    boxes = r.boxes.xyxy.cpu().numpy() if r.boxes.xyxy is not None else np.zeros((0,4))
    confs = r.boxes.conf.cpu().numpy()  if r.boxes.conf is not None else np.zeros((0,))
    clses = r.boxes.cls.cpu().numpy()   if r.boxes.cls  is not None else np.zeros((0,))
    names = r.names

    dets = []
    for b, c, k in zip(boxes, confs, clses):
        nm = names[int(k)]
        if WANTED and nm not in WANTED:
            continue
        ap = bbox_area_pct(b, W, H)
        if ap < AREA_MIN_PCT or ap > AREA_MAX_PCT:
            continue
        if float(c) < CONF_THRES:
            continue
        dets.append((nm, [float(b[0]), float(b[1]), float(b[2]), float(b[3])], float(c)))

    # ---- Tracking generale ----
    # (anche in lock mode continuiamo a tracciare tutto: ci serve per ri-agganciare)
    pairs, un_tr, un_dt = associate_tracks(tracks, dets, ASSOC_IOU_MIN)
    for ti, dj in pairs:
        nm, bx, cf = dets[dj]
        tracks[ti].update(bx, cf)
    for ti in un_tr:
        tracks[ti].mark_miss()
    for dj in un_dt:
        nm, bx, cf = dets[dj]
        tracks.append(Track(nm, bx, cf))
    tracks = [tr for tr in tracks if tr.misses < K_OUT]

    # assegna label ai nuovi confermati che non hanno ancora label
    for tr in tracks:
        if tr.confirmed and tr.label is None:
            class_counters[tr.cls] = class_counters.get(tr.cls, 0) + 1
            tr.label = f"{tr.cls}_{class_counters[tr.cls]}"

    # aggiorna vettore/mappa confermati
    confirmed = [tr for tr in tracks if tr.confirmed]
    labels_correnti = [tr.label for tr in confirmed if tr.label is not None]
    label_to_track = {tr.label: tr for tr in confirmed if tr.label is not None}

    # controllo stabilità inventario (insieme label invariato)
    current_set = set(labels_correnti)
    if current_set == last_label_set:
        stable_counter += 1
    else:
        stable_counter = 0
        last_label_set = current_set

    # ------------- TARGET LOCK -------------
    # Se abbiamo un target selezionato, aggiorniamo il puntatore al track migliore.
    if selected_label is not None:
        # 1) se abbiamo ancora lo stesso oggetto (stessa istanza Track), ok
        # 2) altrimenti, cerchiamo tra i confermati della stessa classe quello con IoU massimo
        cand = None
        best_iou = -1.0
        last_box = selected_lastbox if selected_lastbox is not None else (
            selected_track.box if (selected_track is not None) else None
        )

        # se il track attuale esiste ancora tra "tracks", usalo
        still_there = selected_track in tracks if selected_track is not None else False
        if not still_there:
            # prova a ri-agganciare per classe+IoU
            for tr in confirmed:
                if tr.cls != selected_class:
                    continue
                if last_box is None:
                    cand = tr; break
                iou = iou_xyxy(last_box, tr.box)
                if iou > best_iou:
                    best_iou = iou; cand = tr
            if cand is not None and (best_iou >= LOCK_REACQUIRE_IOU_MIN or last_box is None):
                selected_track = cand
                # IMPORTANTISSIMO: manteniamo lo stesso label scelto dall'utente
                selected_track.label = selected_label
                selected_lastbox = list(selected_track.box)
                lock_miss_count = 0
            else:
                lock_miss_count += 1
        else:
            # track esiste, aggiorna last box
            selected_lastbox = list(selected_track.box)
            lock_miss_count = 0

        # se non riusciamo a ri-agganciare per troppo tempo → avviso
        if lock_miss_count >= LOCK_REACQUIRE_MAX_MISS:
            # mostriamo messaggio ma continuiamo a vivere (potrebbe rientrare)
            pass

    # ------------- DISEGNO -------------
    vis = frame.copy()

    if selected_label is None:
        # non ho ancora selezionato: disegna tutto (pending= giallo; confirmed= verde)
        for tr in tracks:
            x1,y1,x2,y2 = tr.xyxy_int()
            if not tr.confirmed:
                color = palette_pending; thickness = 1
                label = f"{tr.cls} conf:{tr.conf_ema:.2f} h:{tr.hits}"
            else:
                color = palette_confirmed; thickness = 2
                lab = tr.label if tr.label else tr.cls
                label = f"{lab} conf:{tr.conf_ema:.2f}"
            cv2.rectangle(vis, (x1,y1), (x2,y2), color, thickness)
            cv2.putText(vis, label, (x1, max(20, y1-10)), FONT, 0.55, color, 2)

        # prompt per selezione quando inventario è stabile
        if (not selection_mode) and stable_counter >= STABLE_TICKS and len(labels_correnti) > 0:
            prompt = "INVENTARIO STABILE: digita nel terminale il label dell'attrezzo (es. hammer_1)."
            cv2.putText(vis, prompt, (20, 40), FONT, 0.7, (0,0,255), 2)
            cv2.putText(vis, f"Disponibili: {', '.join(labels_correnti)}", (20, 70), FONT, 0.6, (0,0,255), 2)
            selection_mode = True

    else:
        # SOLO il selezionato (se c'è) in rosso, con info sempre aggiornate
        if selected_track is not None and selected_track.confirmed:
            x1,y1,x2,y2 = selected_track.xyxy_int()
            cx, cy = center_of(selected_track.box)
            w = x2 - x1; h = y2 - y1
            cv2.rectangle(vis, (x1,y1), (x2,y2), palette_selected, 3)
            big = f"{selected_label}  CONF:{selected_track.conf_ema:.2f}"
            cv2.putText(vis, big, (max(10, x1), max(30, y1-15)), FONT, 0.8, palette_selected, 2)
            info1 = f"bbox: [{x1},{y1},{x2},{y2}]  w:{w}  h:{h}"
            info2 = f"centroide: ({int(round(cx))}, {int(round(cy))})"
            cv2.putText(vis, info1, (20, H-50), FONT, 0.7, palette_selected, 2)
            cv2.putText(vis, info2, (20, H-20), FONT, 0.7, palette_selected, 2)
        else:
            msg = "Target perso... provo a ri-agganciare."
            cv2.putText(vis, msg, (20, 40), FONT, 0.8, (0,0,255), 2)

    # Mostra finestra
    cv2.imshow(WINDOW_NAME, vis)
    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q')):
        break
    if key in (ord('r'),):  # reset selezione
        selected_label = None
        selected_class = None
        selected_track = None
        selected_lastbox = None
        lock_miss_count = 0
        selection_mode = False
        stable_counter = 0
        print("Reset selezione. Torno all'inventario.")

    # Se devo chiedere in console cosa vuoi, lo faccio quando scatta selection_mode
    if selection_mode and selected_label is None:
        print("\n=== INVENTARIO STABILE ===")
        print("Disponibili:", ", ".join(labels_correnti))
        user = input("Digita il label dell'attrezzo da selezionare (es. hammer_1): ").strip()
        if user:
            # cerca il track per label
            tr = label_to_track.get(user)
            if tr is not None:
                selected_label = user
                selected_class = tr.cls
                selected_track = tr
                selected_lastbox = list(tr.box)
                lock_miss_count = 0
                # stampa una volta le info al momento della selezione
                x1,y1,x2,y2 = tr.xyxy_int()
                cx, cy = center_of(tr.box)
                w = x2 - x1; h = y2 - y1
                print(f"\nSelezionato: {selected_label}")
                print(f"bbox_xyxy = [{x1},{y1},{x2},{y2}]  (w={w}, h={h})")
                print(f"centroide  = ({int(round(cx))}, {int(round(cy))})")
            else:
                print("Label non trovato tra i confermati. Continua l'acquisizione...")
                # reset per riprovare quando torna stabile
                selected_label = None
                selection_mode = False
                stable_counter = 0
        else:
            print("Nessun input. Continua l'acquisizione...")
            selected_label = None
            selection_mode = False
            stable_counter = 0

cap.release()
cv2.destroyAllWindows()
