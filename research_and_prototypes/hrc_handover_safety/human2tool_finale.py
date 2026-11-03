'''questo codice fa parte di una triade che  servono per assicurarsi la presa, da parte del operatore ,del tool in mano al tiago 

segue la seguente logica:

ipotesi: il tiago ha già il tool desiderato 

input: siamo già a conoscenza della classe del tool ( precedentemente  avremo dovuto ricevere il  comnaodo vocale che identifica il tool ) 
       { ovviamnete  non essenso presente un comando vocale ancora, all'inizio dobbiamo da terminale selezionare il tool che il tiago ha in mano. questa scelta è stata fatta per evitare inutilmente di cercare altre classi }

riconoscimneto : la presa avviene se si verificano le segunti condizioni 
1) la mano\ dita sono dentro la box del tool 
2) la mano è chiusa

oss: -sono state definite due tipologie di chiusura della mano. una di precisione utile per atrezzi sottili e una di potenza per oggetti grossi 
     - potrebbe essere che la pinza del tiago o la mano del opeartore durante la presa coprino e nascondio il tool ergo il codice potrebbe non funzonare dato che non riconoscerebbe piu la box.
       Per questo è stata creata  una roi sostitutiva alla box dopo 5 frame che è presneta il tool  

OUTPUT CODICE: si stampa un mesaggio di conferma presa ( che sara in nostro booleano da mandare ad altri codici che in modo parallelo  verificano l'esecuzione di:  questo codice + conferma dovuto a sensore di torsione + conferma vocale  ==> la pinza del tiago si apre  )       
 

'''
import time
import math
import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np

import torch
try:
    torch.cuda.init()
except Exception as e:
    print(f"\n❌ ERRORE NASCOSTO: {e}")
    
# ================= CONFIG BASE =================
WEIGHTS_PATH = r"C:\Users\ronan\OneDrive\Documenti\progetti_python\01_vision_ai\IRIM_Challenge2025\codici python_visual_recognition\pesi_tool\prova2.pt"
DEVICE       = "0"           # metti 0 se hai GPU CUDA
CONF_THRES   = 0.50            # Confidenza per la creazione della ROI
IOU_THRES    = 0.35
WANTED       = { "drill","hammer","pliers","screwdriver","wrench"}

def it_name(nm):
    return {

        "drill":"trapano",
        "hammer":"martello",
        "pliers":"pinza",
        "screwdriver":"cacciavite",
        "wrench":"chiave",
       

    }.get(nm,nm)

FLIP_VIEW    = True            # vista specchio (più naturale in debug)

# Colori GUI
COLOR_GREEN  = (0, 200, 0)
COLOR_RED    = (0, 0, 255)
COLOR_BOX    = (0, 255, 0)
COLOR_ROI    = (255, 0, 0)    


# ====== Soglie mano (MediaPipe) ======
THRESH_FINGER_CLOSE = 0.60     # mano "chiusa" 

# ====== Filtro area box ======
AREA_MIN_FRAC_GLOBAL = 0.005
AREA_MAX_FRAC_GLOBAL = 0.70

# ====== Precision (pinch + dito vicino al bordo tool) ======
TAU_PINCH_ENTER = 0.45         
TAU_PINCH_EXIT  = 0.55        
TAU_BORDO_ENTER = 0.35         
TAU_BORDO_EXIT  = 0.45

# ====== Power (fallback): mano chiusa + ≥2 dita nel box ======
MIN_FINGERS_IN  = 2

# ====== Isteresi temporale (frame-based) ======
N_GRAB_PREC     = 2
N_RELEASE_PREC  = 4
N_GRAB_POWER    = 3
N_RELEASE_POWER = 6

# Slender mode
AR_SLENDER        = 2.2

# ====== Temporal smoothing tool (anti glitch) ======

PERSIST_MIN_VOTES  = 3          # quanti frame coerenti servono per "accettare" il tool

# ====== ROI gating per YOLO  ======
ROI_MARGIN_FRAC     = 0.10      # margine attorno al box tool 
ROI_DRAW            = True      # disegna il rettangolo ROI 

# ================= INIZIO MODELLI =================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam non disponibile")

# YOLO
model = YOLO(WEIGHTS_PATH)
model.overrides["conf"] = CONF_THRES
model.overrides["iou"]  = IOU_THRES

available = set(model.names.values())
WANTED = WANTED & available

# ================== INPUT SCELTA ==================
print("Seleziona il tool da cercare:")
print("1 - trapano")
print("2 - martello")
print("3 - pinza")
print("4 - cacciavite")
print("5 - chiave")

selected = None
while True:
    choice = input("Seleziona 1 (trapano) o 2 (martello) o 3 (pinza) o 4 (cacciavite) o 5(chiave): ").strip()
    if choice == "2":
        selected = "hammer"; print("Hai selezionato: Martello"); break
    
    elif choice == "4":
        selected = "screwdriver"; print("Hai selezionato: Cacciavite"); break
    
    elif choice == "1":
        selected = "drill"; print("Hai selezionato: trapano"); break
    
    elif choice == "5":
        selected = "wrench"; print("Hai selezionato: chiave"); break
    
    elif choice == "3":
        selected = "pliers"; print("Hai selezionato: pinza"); break
    else:
        print("Strumento non disponibile. Riprova.")
        

# Mappa classe selezionata 
model_names = getattr(model, "names", None)
if model_names is None:
    model_names = getattr(getattr(model, "model", None), "names", None)
if isinstance(model_names, dict):
    selected_id = [k for k, v in model_names.items() if v == selected][0]
else:
    selected_id = [i for i, v in enumerate(model_names) if v == selected][0]
model.overrides["classes"] = [selected_id]  

# MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.7,
    max_num_hands=2
)

time.sleep(0.1)

# Stato per mano (contatori di isteresi)
hand_states = {
    'Left':  {'prec_grab':0,'prec_rel':0,'pow_grab':0,'pow_rel':0,'is_prec':False,'is_pow':False},
    'Right': {'prec_grab':0,'prec_rel':0,'pow_grab':0,'pow_rel':0,'is_prec':False,'is_pow':False},
}

# Tracking semplice del tool (voti + IoU)
tool_track = {'name': None, 'box': None, 'votes': 0, 'frame_count': 0}

# ROI state
roi_box = None                



# Indici
PALM_SET   = [0, 5, 9, 13, 17]
FOUR_TIPS  = [8, 12, 16, 20]
ALL_TIPS   = [4, 8, 12, 16, 20]
LABEL_TO_IT = {'Left': 'SX', 'Right': 'DX'}

# ==================== UTILS ====================



def l2(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])



def point_in_box(px, py, box):
    x1, y1, x2, y2 = box
    return (px >= x1) and (px <= x2) and (py >= y1) and (py <= y2)

def min_dist_point_to_box(px, py, box):
    x1, y1, x2, y2 = box
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    dx = 0 if x1 <= px <= x2 else (x1 - px if px < x1 else px - x2)
    dy = 0 if y1 <= py <= y2 else (y1 - py if py < y1 else py - y2)
    return math.hypot(dx, dy)

def iou_boxes(b1, b2):
    x1, y1, x2, y2 = b1
    X1, Y1, X2, Y2 = b2
    xi1, yi1 = max(x1, X1), max(y1, Y1)
    xi2, yi2 = min(x2, X2), min(y2, Y2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter == 0: return 0.0
    a1 = max(0, x2 - x1) * max(0, y2 - y1)
    a2 = max(0, X2 - X1) * max(0, Y2 - Y1)
    return inter / max(1.0, a1 + a2 - inter)

def clamp_box(box, w, h):
    x1, y1, x2, y2 = box
    return [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]

def expand_box(box, margin_frac, w, h, scale=1.0):
    """Espande il box con un margine frazionale e una scala globale, clampando ai limiti frame."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = (x2 - x1) * (1.0 + 2.0 * margin_frac) * scale
    bh = (y2 - y1) * (1.0 + 2.0 * margin_frac) * scale
    nx1 = int(cx - bw / 2.0)
    ny1 = int(cy - bh / 2.0)
    nx2 = int(cx + bw / 2.0)
    ny2 = int(cy + bh / 2.0)
    return clamp_box([nx1, ny1, nx2, ny2], w, h)

def palm_center_and_scale(landmarks, w, h):
    pts = [(int(landmarks[idx].x*w), int(landmarks[idx].y*h)) for idx in PALM_SET]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    wrist = (int(landmarks[0].x*w), int(landmarks[0].y*h))
    mcp9  = (int(landmarks[9].x*w), int(landmarks[9].y*h))
    scale = max(1.0, l2(wrist, mcp9))
    return (cx, cy), scale

def four_finger_closure(landmarks, w, h, palm_center, scale):
    acc = 0.0
    for idx in FOUR_TIPS:
        tip = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
        acc += l2(tip, palm_center) / scale
    return acc / len(FOUR_TIPS)

def get_tip_points(landmarks, w, h, indices):
    return [(int(landmarks[idx].x*w), int(landmarks[idx].y*h)) for idx in indices]

# ==================== LOOP ====================
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if FLIP_VIEW:
        frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ----------------- MediaPipe Hands -----------------
    res = hands.process(rgb)
    annotated = frame.copy()

    hands_info = []
    detected_labels = set()

    if res.multi_hand_landmarks:
        for i, handLms in enumerate(res.multi_hand_landmarks):
            # Skeleton ON (utile in debug handover)
            mp_draw.draw_landmarks(annotated, handLms, mp_hands.HAND_CONNECTIONS)

            label_raw = None
            if res.multi_handedness and len(res.multi_handedness) > i:
                label_raw = res.multi_handedness[i].classification[0].label  # 'Left'/'Right'
            label = label_raw or 'Right'

            palm_c, hand_scale = palm_center_and_scale(handLms.landmark, w, h)
            mean_close = four_finger_closure(handLms.landmark, w, h, palm_c, hand_scale)
            tips_all   = get_tip_points(handLms.landmark, w, h, ALL_TIPS)
            visible_tips = sum(1 for (px, py) in tips_all if 0 <= px < w and 0 <= py < h)

            hands_info.append({
                'label': label,
                'scale': hand_scale,
                'mean_close': mean_close,
                'tips_all': tips_all,
                'visible_tips': visible_tips
            })
            detected_labels.add(label)

    # ----------------- YOLO con ROI gating -----------------
    yolo_frame = frame
    roi_offset = (0, 0)

    # Se il tool è stabile, aggiorna la ROI
    if tool_track['votes'] >= PERSIST_MIN_VOTES:
        roi_box = expand_box(tool_track['box'], ROI_MARGIN_FRAC, w, h)
        x1r, y1r, x2r, y2r = roi_box
        yolo_frame = frame[y1r:y2r, x1r:x2r]
        roi_offset = (x1r, y1r)

    # Esegui YOLO sul frame
    yres = model.predict(source=yolo_frame, verbose=False, device=DEVICE)
    sel_name, sel_box, sel_conf = None, None, -1.0

    # Se troviamo un oggetto con confidenza alta
    if yres and len(yres) > 0 and yres[0].boxes is not None and yres[0].boxes.xyxy is not None:
        names = yres[0].names
        ox, oy = roi_offset

        for box, cls_id, conf in zip(yres[0].boxes.xyxy, yres[0].boxes.cls, yres[0].boxes.conf):
            name = names[int(cls_id)]
            if WANTED and name not in WANTED:
                continue

            x1, y1, x2, y2 = map(int, box.tolist())
            x1 += ox; y1 += oy; x2 += ox; y2 += oy


            area_frac = ((x2 - x1) * (y2 - y1)) / max(1.0, (w * h))
            if AREA_MIN_FRAC_GLOBAL <= area_frac <= AREA_MAX_FRAC_GLOBAL and conf > sel_conf:
                sel_name, sel_box, sel_conf = name, [x1, y1, x2, y2], conf

            

    # ----------------- Stabilizza l'oggetto trovato -----------------
    if sel_box is not None:
        tool_track['box'] = sel_box
        tool_track['votes'] = min(tool_track['votes'] + 1, PERSIST_MIN_VOTES)
        tool_track['frame_count'] += 1  # Incrementa il conteggio dei frame
    else:
        tool_track['votes'] = max(0, tool_track['votes'] - 1)
        tool_track['frame_count'] = 0  # Resetta il conteggio dei frame

    # Aggiorna stati per mani non visibili
    for lbl in hand_states.keys():
        if lbl not in detected_labels:
            st = hand_states[lbl]
            st['prec_rel'] += 1
            st['pow_rel']  += 1
            st['prec_grab'] = 0
            st['pow_grab']  = 0
            if st['is_prec'] and st['prec_rel'] >= N_RELEASE_PREC:
                st['is_prec'] = False
            if st['is_pow'] and st['pow_rel'] >= N_RELEASE_POWER:
                st['is_pow'] = False

    # Decisione (solo booleani) + stato aggregato
    grasping_labels = []

    if tool_track['box'] is not None and len(hands_info) > 0:
        x1, y1, x2, y2 = tool_track['box']
        tool_w  = max(1, x2 - x1)
        tool_h  = max(1, y2 - y1)
        long_side  = max(tool_w, tool_h)
        short_side = max(1, min(tool_w, tool_h))
        ar_tool = long_side / short_side

        for info in hands_info:
            lbl        = info['label']
            hand_scale = info['scale']
            mean_close = info['mean_close']
            tips_all   = info['tips_all']
            st         = hand_states[lbl]

            # pinch (pollice 4, indice 8)
            thumb_tip = tips_all[0]; index_tip = tips_all[1]
            d_pinch_norm = l2(thumb_tip, index_tip) / hand_scale
            pinch_ok_enter = (d_pinch_norm < TAU_PINCH_ENTER)
            pinch_ok_exit  = (d_pinch_norm < TAU_PINCH_EXIT)
            pinch_ok = pinch_ok_enter if not st['is_prec'] else pinch_ok_exit

            # min distanza dita -> bordo box + contatore dita dentro
            dmin_px = float('inf'); n_in = 0
            for (px, py) in tips_all:
                if point_in_box(px, py, tool_track['box']):
                    n_in += 1
                    d = 0.0
                else:
                    d = min_dist_point_to_box(px, py, tool_track['box'])
                if d < dmin_px: dmin_px = d
            d_bordo_norm = dmin_px / hand_scale
            bordo_ok_enter = (d_bordo_norm < TAU_BORDO_ENTER)
            bordo_ok_exit  = (d_bordo_norm < TAU_BORDO_EXIT)
            bordo_ok = bordo_ok_enter if not st['is_prec'] else bordo_ok_exit

            n_in_ok   = (n_in >= MIN_FINGERS_IN)
            closed_ok = (mean_close < THRESH_FINGER_CLOSE)

            # Vie
            prec_now = (pinch_ok and bordo_ok)
            pow_now  = (n_in_ok and closed_ok)

            # Slender => privilegia precisione
            if ar_tool > AR_SLENDER:
                pow_now = False

            # Isteresi temporale
            if prec_now:
                st['prec_grab'] += 1; st['prec_rel'] = 0
            else:
                st['prec_rel']  += 1; st['prec_grab'] = 0

            if pow_now:
                st['pow_grab']  += 1; st['pow_rel'] = 0
            else:
                st['pow_rel']   += 1; st['pow_grab'] = 0

            if not st['is_prec'] and st['prec_grab'] >= N_GRAB_PREC:
                st['is_prec'] = True
            elif st['is_prec'] and st['prec_rel'] >= N_RELEASE_PREC:
                st['is_prec'] = False

            if not st['is_pow'] and st['pow_grab'] >= N_GRAB_POWER:
                st['is_pow'] = True
            elif st['is_pow'] and st['pow_rel'] >= N_RELEASE_POWER:
                st['is_pow'] = False

            if st['is_prec'] or st['is_pow']:
                grasping_labels.append(lbl)

    # Overlay minimale: solo stato finale
    show = frame.copy()
    if len(grasping_labels) == 0:
        txt, col = "NO GRASP", COLOR_RED
    else:
        it_labels = "+".join(LABEL_TO_IT[lbl] for lbl in grasping_labels)
        # Mostra anche il tipo (PREC/POW) per chiarezza
        kinds = []
        for lbl in grasping_labels:
            st = hand_states[lbl]
            if st['is_prec']: kinds.append(f"{LABEL_TO_IT[lbl]}-PREC")
            if st['is_pow'] : kinds.append(f"{LABEL_TO_IT[lbl]}-POW")
        kind_txt = " | ".join(kinds) if kinds else it_labels
        txt, col = f"GRASP: {it_labels}  ({kind_txt})", COLOR_GREEN
        print(f"HO PRESO :{it_name(selected)} ")
        break

    # Mostra il risultato finale
    cv2.putText(show, txt, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2, cv2.LINE_AA)

    # Disegnare il box dell'oggetto trovato
    if sel_box is not None:
        x1, y1, x2, y2 = sel_box
        cv2.rectangle(show, (x1, y1), (x2, y2), COLOR_BOX, 2)
        cv2.putText(annotated, f"{name} {float(conf):.2f}", (x1, max(15, y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BOX, 2)

    # Disegnare la ROI
    if roi_box is not None and ROI_DRAW:
        x1, y1, x2, y2 = roi_box
        cv2.rectangle(show, (x1, y1), (x2, y2), COLOR_ROI, 2)
        

    # Disegnare le mani
    if res.multi_hand_landmarks:
        for handLms in res.multi_hand_landmarks:
            mp_draw.draw_landmarks(show, handLms, mp_hands.HAND_CONNECTIONS)

    # Mostra il frame
    cv2.imshow("Grasp Detected", show)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
if cap.isOpened():
    hands.close()
    cap.release()
cv2.destroyAllWindows()