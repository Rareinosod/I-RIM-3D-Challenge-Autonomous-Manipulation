# grasp_hammer_mediapipe_yolo_twohands_4fingers.py
import time
import math
import cv2
import mediapipe as mp
from ultralytics import YOLO

# ============ CONFIG ============
WEIGHTS_PATH = "best.pt"                  # <-- il tuo file pesi YOLO
DEVICE       = "cpu"                      # "cpu" oppure 0 per GPU
CONF_THRES   = 0.40
IOU_THRES    = 0.50
WANTED       = {"hammer", "screwdriver"}


FLIP_VIEW          = True                 # specchio webcam
HAND_LABEL_MIRROR  = False                 # se True e FLIP_VIEW=True, inverti Left/Right per avere DX/SX reali
DRAW_DEBUG         = True

# ---- Soglie chiusura mano (nuova metrica: 4 dita, senza pollice) ----
THRESH_FINGER_CLOSE = 0.70               

# ---- Soglie prossimità martello (2a & 2b) ----
BOX_EXPAND_RATIO     = 0.10               # 2a: palmo deve cadere nel box espanso (20%)
K_CENTER_TOOLSIDE    = 0.40               # 2b: d(palmo, centro_tool) < k * lato_lungo_box

# ---- Isteresi (tunable) ----
N_GRAB    = 4  # frame consecutivi per confermare GRASP
N_RELEASE = 6  # frame consecutivi per confermare RELEASE

# ============ INIZIO ============
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam non disponibile")



# ===== CARICO UNA SOLA VOLTA =====
model = YOLO(WEIGHTS_PATH)
# setto conf/iou
model.overrides["conf"] = CONF_THRES
model.overrides["iou"]  = IOU_THRES

# controllo classi mancanti
available = set(model.names.values())
missing   = WANTED - available
if missing:
    print(f"[ATTENZIONE] classi richieste non presenti nei pesi: {sorted(missing)}")

WANTED = WANTED & available
print(f"[INFO] Userò WANTED = {sorted(WANTED)}")


# MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(min_detection_confidence=0.5,
                          min_tracking_confidence=0.7)

time.sleep(0.5)

# Stato grasp PER MANO (SX/DX)
hand_states = {
    'Left':  {'grasp_counter': 0, 'release_counter': 0, 'is_grasping': False},
    'Right': {'grasp_counter': 0, 'release_counter': 0, 'is_grasping': False},
}

# Indici utili
PALM_SET = [0, 5, 9, 13, 17]      # per centro palmo
FOUR_TIPS = [8, 12, 16, 20]       # indice, medio, anulare, mignolo (no pollice)

LABEL_TO_IT = {'Left': 'SX', 'Right': 'DX'}

# ==================== UTILS ====================
def l2(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def point_in_expanded_box(px, py, box, expand_ratio=0.2):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    ex = int(w * expand_ratio / 2)
    ey = int(h * expand_ratio / 2)
    x1e, y1e = x1 - ex, y1 - ey
    x2e, y2e = x2 + ex, y2 + ey
    return (px >= x1e) and (px <= x2e) and (py >= y1e) and (py <= y2e)

def palm_center_and_scale(landmarks, w, h):
    """Centro palmo (media di PALM_SET) e scala mano (distanza wrist(0) - MCP medio(9))."""
    pts = []
    for idx in PALM_SET:
        lm = landmarks[idx]
        pts.append((int(lm.x*w), int(lm.y*h)))
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)

    wrist = (int(landmarks[0].x*w), int(landmarks[0].y*h))
    mcp9  = (int(landmarks[9].x*w), int(landmarks[9].y*h))
    scale = max(1.0, l2(wrist, mcp9))  # evita divisione per 0
    return (cx, cy), scale

def four_finger_closure(landmarks, w, h, palm_center, scale):
    """
    Metrica di chiusura: media delle distanze normalizzate
    delle 4 dita (indice, medio, anulare, mignolo) dal centro palmo.
    Esclude il pollice.
    Restituisce: (media_normalizzata, dizionario singole distanze)
    """
    vals = {}
    acc = 0.0
    for idx in FOUR_TIPS:
        tip = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
        d = l2(tip, palm_center) / scale
        vals[idx] = d
        acc += d
    mean_val = acc / len(FOUR_TIPS)
    return mean_val, vals

def map_label_for_mirror(label):
    # MediaPipe fornisce 'Left'/'Right' riferite all'immagine di input.
    # Se l'immagine è flippata e vogliamo DX/SX reali, invertiamo.
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

    # ----- MediaPipe Hands -----
    res = hands.process(rgb)
    annotated = frame.copy()

    # Raccogli info per TUTTE le mani
    hands_info = []   # [{'label','palm_c','scale','mean_close','close_parts','landmarks'}]
    detected_labels_this_frame = set()

    if res.multi_hand_landmarks:
        # multi_handedness è allineato a multi_hand_landmarks
        for i, handLms in enumerate(res.multi_hand_landmarks):
            # Disegno
            mp_draw.draw_landmarks(annotated, handLms, mp_hands.HAND_CONNECTIONS)

            # Etichetta mano
            label_raw = None
            if res.multi_handedness and len(res.multi_handedness) > i:
                label_raw = res.multi_handedness[i].classification[0].label  # 'Left'/'Right'
            label = map_label_for_mirror(label_raw or 'Right')

            # Centro palmo + scala
            palm_c, hand_scale = palm_center_and_scale(handLms.landmark, w, h)

            # Nuova metrica di chiusura (4 dita, senza pollice)
            mean_close, close_parts = four_finger_closure(handLms.landmark, w, h, palm_c, hand_scale)

            hands_info.append({
                'label': label,
                'palm_c': palm_c,
                'scale': hand_scale,
                'mean_close': mean_close,
                'close_parts': close_parts,  # mappa {8:..,12:..,16:..,20:..}
                'landmarks': handLms
            })
            detected_labels_this_frame.add(label)

            if DRAW_DEBUG:
                cv2.circle(annotated, (int(palm_c[0]), int(palm_c[1])), 6, (255, 255, 0), -1)
                # mostro media e (facoltativo) la sola media; se vuoi i 4 valori, aggiungili
                cv2.putText(
                    annotated,
                    f"{LABEL_TO_IT[label]} close4={mean_close:.2f}",
                    (int(palm_c[0])+8, int(palm_c[1])-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 220, 220), 2
                )

    # ----- YOLO (scegliamo il tool migliore tra le classi WANTED) -----
    yres = model.predict(source=frame, verbose=False, device=DEVICE)
    tool_box = None
    tool_name = None
    tool_conf = None

    if yres and len(yres) > 0 and yres[0].boxes is not None and yres[0].boxes.xyxy is not None:
        names = yres[0].names

        # selezione del box "wanted" a conf più alta
        best_conf = -1.0
        for box, cls_id, conf in zip(yres[0].boxes.xyxy, yres[0].boxes.cls, yres[0].boxes.conf):
            name = names[int(cls_id)]
            if WANTED and name not in WANTED:
                continue
            if float(conf) > best_conf:
                best_conf = float(conf)
                tool_box = list(map(int, box.tolist()))
                tool_name = name
                tool_conf = float(conf)

        # disegno TUTTI i box "wanted"
        for box, cls_id, conf in zip(yres[0].boxes.xyxy, yres[0].boxes.cls, yres[0].boxes.conf):
            name = names[int(cls_id)]
            if WANTED and name not in WANTED:
                continue
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{name} {float(conf):.2f}",
                        (x1, max(15, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # ----- Aggiorna counters per mani NON visibili in questo frame -----
    for lbl in hand_states.keys():
        if lbl not in detected_labels_this_frame:
            hand_states[lbl]['release_counter'] += 1
            hand_states[lbl]['grasp_counter'] = 0
            if hand_states[lbl]['is_grasping'] and hand_states[lbl]['release_counter'] >= N_RELEASE:
                hand_states[lbl]['is_grasping'] = False

    # ----- Doppia conferma GRASP per OGNI MANO (2a AND 2b) -----
    debug_lines = []
    if tool_box is not None:
        x1, y1, x2, y2 = tool_box
        tool_cx = (x1 + x2) // 2
        tool_cy = (y1 + y2) // 2
        tool_w  = max(1, x2 - x1)
        tool_h  = max(1, y2 - y1)
        long_side = max(tool_w, tool_h)

        if DRAW_DEBUG:
            # box espanso (debug)
            ex = int(tool_w * BOX_EXPAND_RATIO / 2)
            ey = int(tool_h * BOX_EXPAND_RATIO / 2)
            cv2.rectangle(annotated, (x1 - ex, y1 - ey), (x2 + ex, y2 + ey), (0, 180, 255), 1)
            cv2.circle(annotated, (tool_cx, tool_cy), 5, (0, 200, 255), -1)

        for info in hands_info:
            lbl        = info['label']
            palm_c     = info['palm_c']
            mean_close = info['mean_close']

            # 1) mano chiusa ( metrica 4-dita senza pollice)
            hand_closed = (mean_close < THRESH_FINGER_CLOSE)

            # 2a) palmo dentro box espanso
            palm_in_box = point_in_expanded_box(int(palm_c[0]), int(palm_c[1]),
                                                tool_box, BOX_EXPAND_RATIO)

            # 2b) distanza palmo->centro tool in PIXEL < k * lato_lungo_box
            d_palm_tool_px = l2(palm_c, (tool_cx, tool_cy))
            cond_center = (d_palm_tool_px < K_CENTER_TOOLSIDE * long_side)

            near_tool = palm_in_box and cond_center
            grasp_condition = hand_closed and near_tool

            # isteresi per questa mano
            if grasp_condition:
                hand_states[lbl]['grasp_counter'] += 1
                hand_states[lbl]['release_counter'] = 0
            else:
                hand_states[lbl]['release_counter'] += 1
                hand_states[lbl]['grasp_counter'] = 0

            if not hand_states[lbl]['is_grasping'] and hand_states[lbl]['grasp_counter'] >= N_GRAB:
                hand_states[lbl]['is_grasping'] = True
            elif hand_states[lbl]['is_grasping'] and hand_states[lbl]['release_counter'] >= N_RELEASE:
                hand_states[lbl]['is_grasping'] = False

            if DRAW_DEBUG:
                debug_lines.append(
                    f"{LABEL_TO_IT[lbl]} closed={hand_closed}  near(2a&2b)={near_tool}  "
                    f"d_px={d_palm_tool_px:.1f}/{K_CENTER_TOOLSIDE*long_side:.1f}"
                )

    # ----- Overlay stato aggregato -----
    grasping_labels = [lbl for lbl, st in hand_states.items() if st['is_grasping']]
    if len(grasping_labels) == 0:
        state_txt = "NO GRASP"
        state_col = (0, 0, 255)
    else:
        it_labels = "+".join(LABEL_TO_IT[lbl] for lbl in grasping_labels)
        state_txt = f"GRASP: {it_labels}"
        state_col = (0, 200, 0)

    cv2.putText(annotated, state_txt, (10, h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_col, 3)

    if DRAW_DEBUG and debug_lines:
        y0 = 90
        for line in debug_lines[:3]:
            cv2.putText(annotated, line, (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 2)
            y0 += 22

    cv2.imshow("Grasp detection (Two Hands + YOLO, 4-finger closure)", annotated)
    k = cv2.waitKey(1) & 0xFF
    if k in (27, ord('q')):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
