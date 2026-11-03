import time
import math
import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import torch

# ==============================================================
# Ocelot v4.3 — Grasp umano su tool
# - Precision-first + Power-fallback con isteresi
# - ROI gating per YOLO (crop attorno al tool) + recovery
# - Uscita latched (persistenza/hold/cooldown) per integrazione con ROS
# - Depth hook opzionale (RGB-D) pronto per il futuro
# ==============================================================

# ================= CONFIG BASE =================
WEIGHTS_PATH = r"IRIM_Challenge2025\codici python_visual_recognition\pesi_tool\best.pt"
DEVICE       = "0"           # metti 0 se hai GPU CUDA
CONF_THRES   = 0.40
IOU_THRES    = 0.35
WANTED       = {"hammer", "screwdriver"}

FLIP_VIEW    = True            # vista specchio (più naturale in debug)

# Colori GUI
COLOR_GREEN  = (0, 200, 0)
COLOR_RED    = (0, 0, 255)
COLOR_BOX    = (0, 255, 0)
COLOR_DEBUG  = (0, 180, 255)

# ====== Soglie mano (MediaPipe) ======
THRESH_FINGER_CLOSE = 0.60     # mano "chiusa" (4 dita vs palmo)  <-- per Power

# ====== Filtro area box ======
# Fallback globale (utile per classi nuove non ancora tarate)
AREA_MIN_FRAC_GLOBAL = 0.005
AREA_MAX_FRAC_GLOBAL = 0.70

# Per-classe (override più stretti, oggi hammer/screwdriver)
CLASS_AREA_FRAC = {
    "hammer":      (0.010, 0.25),
    "screwdriver": (0.004, 0.15),
}

# ====== Precision (pinch + dito vicino al bordo tool) ======
TAU_PINCH_ENTER = 0.45         # d(thumb-index)/hand_scale < 0.45 => entra
TAU_PINCH_EXIT  = 0.55         # >0.55 => esce (isteresi)
TAU_BORDO_ENTER = 0.35         # min_dist(tip→bordo)/hand_scale < 0.35 => vicino
TAU_BORDO_EXIT  = 0.45

# ====== Power (fallback): mano chiusa + ≥2 dita nel box ======
MIN_FINGERS_IN  = 2

# ====== Isteresi temporale (frame-based) ======
N_GRAB_PREC     = 2
N_RELEASE_PREC  = 4
N_GRAB_POWER    = 3
N_RELEASE_POWER = 6

# ====== Gating Power (quando provarla) ======
AR_SLENDER         = 2.0        # aspect ratio > 2.0 => tool "slender" => evita Power
MIN_TIPS_FOR_PREC  = 3          # se <3 tip visibili => attiva Power come fallback

# ====== Temporal smoothing tool (anti glitch) ======
PERSIST_IOU_THR    = 0.30       # quanto devono somigliarsi i box tra frame
PERSIST_MIN_VOTES  = 3          # quanti frame coerenti servono per "accettare" il tool

# ====== ROI gating per YOLO (crop + recovery) ======
ROI_MARGIN_FRAC     = 0.40      # margine attorno al box tool per fare il crop (40% per lato)
ROI_RECOVERY_STEPS  = [1.0, 1.5, 2.0]  # scale progressive se non troviamo il tool
ROI_MAX_BACKOFF     = 12        # dopo tot frame falliti, torna a full frame
ROI_DRAW            = True      # disegna il rettangolo ROI (debug)

# ====== LATCH/OUTPUT (per integrazione con ROS) ======
# human_grasp_latched = 1 se la presa è stabile per almeno T_VISION_MS
# poi manteniamo un HOLD e applichiamo un COOLDOWN dopo il rilascio
T_VISION_MS   = 250             # persistenza minima della presa (vision-only)
T_HOLD_MS     = 150             # tempo minimo di mantenimento del latched
T_COOLDOWN_MS = 500             # inibizione post-rilascio per evitare rimbalzi
PUBLISH_STDOUT = True           # stampa cambi di stato su stdout (per log/bridge)

# ====== (OPZ) Depth gating — pronto per RGB-D ======
USE_DEPTH = False               # lascia False su Mac; metti True con RGB-D allineata
DEPTH_MAX_CONTACT_DIST_M = 0.08 # dita↔tool distanza 3D massima (8 cm)
DEPTH_MAX_Z_MISALIGN_M   = 0.06 # differenza di Z massima (6 cm)

# ================= INIZIO MODELLI =================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam non disponibile")

# YOLO
model = YOLO(WEIGHTS_PATH)

# 2. FORZA IL TRASLOCO SULLA RTX 5070 Ti (Aggiungi questa riga!)
model.to('cuda')

model.overrides["conf"] = CONF_THRES
model.overrides["iou"]  = IOU_THRES

# === INIZIO CONTROLLO GPU (Incolla qui!) ===
print("\n" + "="*30)
print("   VERIFICA HARDWARE")
print("="*30)
cuda_disponibile = torch.cuda.is_available()
print(f"CUDA rilevato: {cuda_disponibile}")

if cuda_disponibile:
    nome_gpu = torch.cuda.get_device_name(0)
    print(f"Scheda Video in uso: {nome_gpu}")
    # Controlliamo se YOLO ha effettivamente caricato i pesi sulla GPU
    device_yolo = model.device
    print(f"YOLO sta usando: {device_yolo}")
    
    if "cuda" in str(device_yolo):
        print("🚀 ESITO: La tua RTX 5070 Ti sta spingendo!")
    else:
        print("⚠️ ESITO: CUDA c'è, ma YOLO è finito sulla CPU. Controlla DEVICE='0'")
else:
    print("❌ ESITO: La GPU non è vista. Il PC sta usando l'Intel Ultra 9.")
print("="*30 + "\n")

# === FINE CONTROLLO GPU ===
available = set(model.names.values())
WANTED = WANTED & available

# MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.7,
    max_num_hands=2
)

time.sleep(0.15)

# Stato per mano (contatori di isteresi)
hand_states = {
    'Left':  {'prec_grab':0,'prec_rel':0,'pow_grab':0,'pow_rel':0,'is_prec':False,'is_pow':False},
    'Right': {'prec_grab':0,'prec_rel':0,'pow_grab':0,'pow_rel':0,'is_prec':False,'is_pow':False},
}

# Tracking semplice del tool (voti + IoU)
tool_track = {'name': None, 'box': None, 'votes': 0}

# ROI state
roi_box = None                 # ROI corrente (x1,y1,x2,y2) in coord frame
roi_fail_count = 0             # contatore fallimenti consecutivi nella ROI

# Latch state
latched = 0                    # 0/1
t_grasp_start_ms = None        # timestamp inizio "presa vista"
t_latch_on_ms    = None        # timestamp quando latched è diventato 1
t_cooldown_until = 0.0         # tempo (epoch ms) fino a cui siamo in cooldown

# Landmarks utili
PALM_SET   = [0, 5, 9, 13, 17]
FOUR_TIPS  = [8, 12, 16, 20]
ALL_TIPS   = [4, 8, 12, 16, 20]
LABEL_TO_IT = {'Left': 'SX', 'Right': 'DX'}

# ==================== UTILS ====================
def now_ms():
    return time.time() * 1000.0

def l2(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

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

def point_in_box(px, py, box):
    x1, y1, x2, y2 = box
    return (x1 <= px <= x2) and (y1 <= py <= y2)

def min_dist_point_to_box(px, py, box):
    x1, y1, x2, y2 = box
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    dx = 0 if x1 <= px <= x2 else (x1 - px if px < x1 else px - x2)
    dy = 0 if y1 <= py <= y2 else (y1 - py if py < y1 else py - y2)
    return math.hypot(dx, dy)

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

# --------- (OPZ) Depth helpers ----------
def median_depth_in_box(depth_map, box):
    x1, y1, x2, y2 = [max(0, int(v)) for v in box]
    roi = depth_map[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    vals = roi.flatten()
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return None
    return float(np.median(vals))

def median_depth_at_points(depth_map, pts, r=2):
    zs = []
    h, w = depth_map.shape[:2]
    for (x, y) in pts:
        x1, y1 = max(0, x - r), max(0, y - r)
        x2, y2 = min(w, x + r + 1), min(h, y + r + 1)
        patch = depth_map[y1:y2, x1:x2]
        vals = patch.flatten()
        vals = vals[np.isfinite(vals)]
        vals = vals[vals > 0]
        if vals.size > 0:
            zs.append(float(np.median(vals)))
    if len(zs) == 0:
        return None
    return float(np.median(np.array(zs)))

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
            visible_tips = sum(1 for (px,py) in tips_all if 0 <= px < w and 0 <= py < h)

            hands_info.append({
                'label': label,
                'scale': hand_scale,
                'mean_close': mean_close,
                'tips_all': tips_all,
                'visible_tips': visible_tips
            })
            detected_labels.add(label)

    # ----------------- YOLO con ROI gating -----------------
    # Strategia:
    # - Se abbiamo un tool stabile (tool_track.votes >= PERSIST_MIN_VOTES), centriamo la ROI su quel box esteso.
    # - Se falliamo a trovare tool dentro la ROI, aumentiamo la scala (ROI_RECOVERY_STEPS).
    # - Se continuiamo a fallire, torniamo a full frame per qualche ciclo e poi riproviamo.
    yolo_frame = frame
    roi_offset = (0, 0)

    have_stable_tool = (tool_track['box'] is not None and tool_track['votes'] >= PERSIST_MIN_VOTES)

    if have_stable_tool:
        scale_idx = min(roi_fail_count, len(ROI_RECOVERY_STEPS)-1)
        scale     = ROI_RECOVERY_STEPS[scale_idx]
        roi_box   = expand_box(tool_track['box'], ROI_MARGIN_FRAC, w, h, scale=scale)
        x1r, y1r, x2r, y2r = roi_box
        yolo_frame = frame[y1r:y2r, x1r:x2r]
        roi_offset = (x1r, y1r)

        if ROI_DRAW:
            cv2.rectangle(annotated, (x1r, y1r), (x2r, y2r), (255, 128, 0), 2)
    else:
        # se non abbiamo ancora un tool stabile e abbiamo fallito troppe volte, fai full frame
        if roi_fail_count >= ROI_MAX_BACKOFF:
            roi_box = None
            yolo_frame = frame
            roi_offset = (0, 0)
        # altrimenti lascia roi_box=None (full frame di default)

    # Esegui YOLO sul frame (ROI o full)
    yres = model.predict(source=yolo_frame, verbose=False, device=DEVICE)
    sel_name, sel_box, sel_conf = None, None, -1.0

    if yres and len(yres) > 0 and yres[0].boxes is not None and yres[0].boxes.xyxy is not None:
        names = yres[0].names
        ox, oy = roi_offset

        for box, cls_id, conf in zip(yres[0].boxes.xyxy, yres[0].boxes.cls, yres[0].boxes.conf):
            name = names[int(cls_id)]
            if WANTED and name not in WANTED:
                continue

            # Trasla le coordinate della ROI alle coordinate del frame
            x1, y1, x2, y2 = map(int, box.tolist())
            x1 += ox; y1 += oy; x2 += ox; y2 += oy

            # Filtro area per classe (o globale)
            area_frac = ((x2 - x1) * (y2 - y1)) / max(1.0, (w * h))
            min_frac, max_frac = CLASS_AREA_FRAC.get(
                name, (AREA_MIN_FRAC_GLOBAL, AREA_MAX_FRAC_GLOBAL)
            )
            if not (min_frac <= area_frac <= max_frac):
                continue

            # Disegna box validi (debug)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_BOX, 2)
            cv2.putText(annotated, f"{name} {float(conf):.2f}", (x1, max(15, y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BOX, 2)

            if float(conf) > sel_conf:
                sel_name, sel_box, sel_conf = name, [x1, y1, x2, y2], float(conf)

    # ----------------- Smoothing tool (voti+IoU) + ROI recovery update -----------------
    use_tool_box = None
    found_this_frame = (sel_box is not None)

    if found_this_frame:
        roi_fail_count = 0  # reset fallimenti ROI (abbiamo trovato qualcosa)
        if tool_track['name'] == sel_name and tool_track['box'] is not None:
            iou = iou_boxes(tool_track['box'], sel_box)
            if iou >= PERSIST_IOU_THR:
                tool_track['box'] = sel_box
                tool_track['votes'] += 1
            else:
                tool_track['name'] = sel_name
                tool_track['box']  = sel_box
                tool_track['votes'] = 1
        else:
            tool_track['name'] = sel_name
            tool_track['box']  = sel_box
            tool_track['votes'] = 1

        if tool_track['votes'] >= PERSIST_MIN_VOTES:
            use_tool_box = tool_track['box']
    else:
        roi_fail_count += 1
        tool_track['votes'] = max(0, tool_track['votes'] - 1)
        if tool_track['votes'] < PERSIST_MIN_VOTES:
            use_tool_box = None
        else:
            use_tool_box = tool_track['box']

    # ----------------- Decadimento stati per mani non visibili -----------------
    for lbl in hand_states.keys():
        if lbl not in detected_labels:
            st = hand_states[lbl]
            st['prec_rel'] += 1; st['pow_rel'] += 1
            st['prec_grab'] = 0; st['pow_grab'] = 0
            if st['is_prec'] and st['prec_rel'] >= N_RELEASE_PREC:
                st['is_prec'] = False
            if st['is_pow'] and st['pow_rel'] >= N_RELEASE_POWER:
                st['is_pow'] = False

    # ----------------- Decisione: Precision-first + Power-fallback -----------------
    grasping_labels = []
    kinds = []

    if use_tool_box is not None and hands_info:
        x1t, y1t, x2t, y2t = use_tool_box
        tool_w = max(1, x2t - x1t); tool_h = max(1, y2t - y1t)
        long_side  = max(tool_w, tool_h)
        short_side = max(1, min(tool_w, tool_h))
        ar_tool = long_side / short_side  # > AR_SLENDER => slender (es. cacciavite)

        # (OPZ) Depth: pre-calcola Z del tool (mediana nel box)
        tool_z = None
        if USE_DEPTH:
            # >>> Inserisci qui la tua depth_map (metri) allineata all'RGB:
            # depth_map = ...
            # tool_z = median_depth_in_box(depth_map, use_tool_box)
            pass

        for info in hands_info:
            lbl  = info['label']
            st   = hand_states[lbl]
            scl  = info['scale']
            tips = info['tips_all']

            # ---------- Precision: pinch + bordo vicino ----------
            d_pinch = l2(tips[0], tips[1]) / scl
            pinch_enter = (d_pinch < TAU_PINCH_ENTER)
            pinch_exit  = (d_pinch < TAU_PINCH_EXIT)
            pinch_ok = pinch_enter if not st['is_prec'] else pinch_exit

            dmin_px = float('inf'); n_in = 0
            for (px, py) in tips:
                if point_in_box(px, py, use_tool_box):
                    n_in += 1
                    d = 0.0
                else:
                    d = min_dist_point_to_box(px, py, use_tool_box)
                if d < dmin_px: dmin_px = d
            d_bordo = dmin_px / scl
            bordo_enter = (d_bordo < TAU_BORDO_ENTER)
            bordo_exit  = (d_bordo < TAU_BORDO_EXIT)
            bordo_ok = bordo_enter if not st['is_prec'] else bordo_exit

            n_in_ok   = (n_in >= MIN_FINGERS_IN)
            closed_ok = (info['mean_close'] < THRESH_FINGER_CLOSE)

            # ---------- (OPZ) Depth gating ----------
            depth_ok = True
            if USE_DEPTH:
                # tips_z = median_depth_at_points(depth_map, tips, r=2)
                # if tool_z is not None and tips_z is not None:
                #     same_plane = abs(tips_z - tool_z) <= DEPTH_MAX_Z_MISALIGN_M
                #     near_3d    = abs(tips_z - tool_z) <= DEPTH_MAX_CONTACT_DIST_M
                #     depth_ok   = same_plane and near_3d
                # else:
                #     depth_ok = False
                pass

            # ---------- Valutazioni correnti ----------
            prec_now = (pinch_ok and bordo_ok and depth_ok)   # Precision prima
            try_power = (not prec_now) and (
                (info['visible_tips'] < MIN_TIPS_FOR_PREC) or (ar_tool <= AR_SLENDER)
            )
            pow_now  = try_power and (n_in_ok and closed_ok and depth_ok)

            # ---------- Isteresi temporale ----------
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

            # ---------- Stato finale per mano ----------
            active = []
            if st['is_prec']: active.append("prec")
            if st['is_pow'] : active.append("pow")
            if active:
                grasping_labels.append(lbl)
                kinds.append(f"{LABEL_TO_IT[lbl]}-" + "|".join(active))

    # ============== LATCH LOGIC (persistenza/hold/cooldown) ==============
    tnow = now_ms()

    # human_grasp_now: almeno una mano in grasp (prec o pow)
    human_grasp_now = (len(grasping_labels) > 0)

    # Cooldown: se in cooldown, non consentire nuovi latch
    in_cooldown = (tnow < t_cooldown_until)

    if human_grasp_now and not in_cooldown:
        # start o prolungamento finestra di visione valida
        if t_grasp_start_ms is None:
            t_grasp_start_ms = tnow
        # se la presa dura abbastanza, settiamo il latch
        if not latched and (tnow - t_grasp_start_ms >= T_VISION_MS):
            latched = 1
            t_latch_on_ms = tnow
            if PUBLISH_STDOUT:
                print("[GRASP_LATCH] 1 (vision persistence OK)")
    else:
        # reset finestra visione
        t_grasp_start_ms = None
        # se eravamo latched, controlla HOLD, poi rilascia e avvia cooldown
        if latched:
            # mantieni per almeno T_HOLD_MS prima di rilasciare
            if (tnow - (t_latch_on_ms or tnow)) >= T_HOLD_MS:
                latched = 0
                t_latch_on_ms = None
                t_cooldown_until = tnow + T_COOLDOWN_MS
                if PUBLISH_STDOUT:
                    print("[GRASP_LATCH] 0 (release -> cooldown)")

    # ----------------- Overlay finale -----------------
    if not grasping_labels:
        txt, col = "NO GRASP", COLOR_RED
    else:
        hands_txt = "+".join(LABEL_TO_IT[l] for l in grasping_labels)
        kind_txt  = " | ".join(kinds) if kinds else hands_txt
        txt, col = f"GRASP: {hands_txt}  ({kind_txt})", COLOR_GREEN

    cv2.putText(annotated, txt, (12, h-48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 3)

    # Stato latched + cooldown (debug integrativo)
    latch_txt = f"LATCH={latched}  COOL={'ON' if in_cooldown else 'OFF'}"
    cv2.putText(annotated, latch_txt, (12, h-16), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (200, 255, 200) if latched else (180, 180, 180), 2)

    # Debug minimale: per mano, quante punte dita sono "visibili"
    y0 = 30
    for info in hands_info:
        cv2.putText(annotated, f"{LABEL_TO_IT[info['label']]} tips={info['visible_tips']}",
                    (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_DEBUG, 2)
        y0 += 22

    cv2.imshow(" ROI YOLO + Latch + Precision/Power (+Depth hook)", annotated)

    k = cv2.waitKey(1) & 0xFF
    if k in (27, ord('q')):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
