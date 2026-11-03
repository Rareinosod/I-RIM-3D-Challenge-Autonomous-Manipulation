import time
import math
import cv2
import mediapipe as mp
from ultralytics import YOLO

# ============ CONFIG ============
WEIGHTS_PATH = r"IRIM_Challenge2025\codici python_visual_recognition\pesi_tool\best.pt"
DEVICE       = "0"                      # usa 0 per GPU se disponibile
CONF_THRES   = 0.40
IOU_THRES    = 0.30
WANTED       = {"hammer", "screwdriver"}

FLIP_VIEW          = True
HAND_LABEL_MIRROR  = False

# Colori
COLOR_GREEN  = (0, 200, 0)
COLOR_RED    = (0, 0, 255)

# ---- Soglie ----
THRESH_FINGER_CLOSE = 0.60      # mano chiusa (4 dita vs palmo)
AREA_MIN_FRAC = 0.005
AREA_MAX_FRAC = 0.60

# Precision / Power
TAU_PINCH_ENTER   = 0.45
TAU_PINCH_EXIT    = 0.55
TAU_BORDO_ENTER   = 0.35
TAU_BORDO_EXIT    = 0.45
MIN_FINGERS_IN    = 2

# Isteresi temporali
N_GRAB_PREC       = 2
N_RELEASE_PREC    = 4
N_GRAB_POWER      = 3
N_RELEASE_POWER   = 6

# Slender mode
AR_SLENDER        = 2.2

# ============ INIZIO ============
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam non disponibile")

# YOLO
model = YOLO(WEIGHTS_PATH)
model.overrides["conf"] = CONF_THRES
model.overrides["iou"]  = IOU_THRES

available = set(model.names.values())
WANTED = WANTED & available

# MediaPipe Hands (no drawing per velocità)
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.7,
    max_num_hands=2
)

time.sleep(0.2)

# Stato grasp PER MANO (SX/DX)
hand_states = {
    'Left':  {'prec_grab':0,'prec_rel':0,'pow_grab':0,'pow_rel':0,
              'is_prec':False,'is_pow':False},
    'Right': {'prec_grab':0,'prec_rel':0,'pow_grab':0,'pow_rel':0,
              'is_prec':False,'is_pow':False},
}

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

def map_label_for_mirror(label):
    if FLIP_VIEW and HAND_LABEL_MIRROR:
        return 'Left' if label == 'Right' else 'Right'
    return label

# ==================== LOOP ====================
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if FLIP_VIEW:
        frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Hands (niente disegno per velocità)
    res = hands.process(rgb)

    hands_info = []
    detected_labels_this_frame = set()

    if res.multi_hand_landmarks:
        for i, handLms in enumerate(res.multi_hand_landmarks):
            label_raw = None
            if res.multi_handedness and len(res.multi_handedness) > i:
                label_raw = res.multi_handedness[i].classification[0].label
            label = map_label_for_mirror(label_raw or 'Right')

            palm_c, hand_scale = palm_center_and_scale(handLms.landmark, w, h)
            mean_close = four_finger_closure(handLms.landmark, w, h, palm_c, hand_scale)
            tips_all  = get_tip_points(handLms.landmark, w, h, ALL_TIPS)

            hands_info.append({
                'label': label,
                'palm_c': palm_c,
                'scale': hand_scale,
                'mean_close': mean_close,
                'landmarks': handLms,
                'tips_all': tips_all
            })
            detected_labels_this_frame.add(label)

    # YOLO (niente disegno box per velocità)
    yres = model.predict(source=frame, verbose=False, device=DEVICE)
    tool_box = None

    if yres and len(yres) > 0 and yres[0].boxes is not None and yres[0].boxes.xyxy is not None:
        names = yres[0].names
        best_conf = -1.0
        for box, cls_id, conf in zip(yres[0].boxes.xyxy, yres[0].boxes.cls, yres[0].boxes.conf):
            name = names[int(cls_id)]
            if WANTED and name not in WANTED:
                continue
            x1, y1, x2, y2 = map(int, box.tolist())
            area_frac = ((x2 - x1) * (y2 - y1)) / max(1.0, (w * h))
            if not (AREA_MIN_FRAC <= area_frac <= AREA_MAX_FRAC):
                continue
            if float(conf) > best_conf:
                best_conf = float(conf)
                tool_box = [x1, y1, x2, y2]

    # Aggiorna stati per mani non visibili
    for lbl in hand_states.keys():
        if lbl not in detected_labels_this_frame:
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

    if tool_box is not None and len(hands_info) > 0:
        x1, y1, x2, y2 = tool_box
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
                if point_in_box(px, py, tool_box):
                    n_in += 1
                    d = 0.0
                else:
                    d = min_dist_point_to_box(px, py, tool_box)
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

    cv2.putText(show, txt, (12, h-16), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 3)
    cv2.imshow("Ocelot v3 - FAST", show)

    k = cv2.waitKey(1) & 0xFF
    if k in (27, ord('q')):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
