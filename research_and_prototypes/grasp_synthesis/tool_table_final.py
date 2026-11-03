''' questo codice serve per riconoscere  un tool desiderato tra tutti quelli  presenti sul tavvolo ed ricavarne la posizioe cartesiana.

INPUT: precederà un commando vocale che identifica il ttol desiderato(non esneos ancora presenta questa feature, si scirve da terminale il tool target)

oss: una volta trovato il tool , deono passare 5 frame di attesa in mod da evitre falsi positivi ed esere sicuri che quello sia il tool cercato

OUTPUT: il codice restitusice l utlimo frame ( immagine statica  ritagliata sulla boxe ) + un file txt con le cordnate del centro della box  e delle sue grandezze.
questo servirà per un secondo codice che in maniera piu precisa riconoscera il punto migliore di presa infatti per ovvi motiiv il centroide della box non seempre corrisponde alla punto di miglir presa. 
come soluzione ci sono due strade 
1) filtro e creo un maschera binaria che mi fa vedere solo il tool 
2) uso la profondità 


versione 2:

devo togliere riconosciento vocale e mettere 
'''

import time
import cv2
import numpy as np
from ultralytics import YOLO

# ================== CONFIGURAZIONE ==================
WEIGHTS_PATH   = "/Users/alessandro/Desktop/IRIM_Challenge2025/codici python_visual_recognition/pesi_tool/prova2.pt"
DEVICE         = "cpu"               # "cpu" oppure 0 per GPU
CLASSES        = {"hammer","pliers","screwdriver","wrench"}  # martello, cacciavite 

# Soglie & filtri
CONF_HAMMER      = 0.45               # conf minima per considerare valido il martello
CONF_SCREWDRIVER = 0.45               # conf minima per considerare valido il cacciavite
CONF_DRILL= 0.45
CONF_PLIERS=0.45
CONF_WRENCH=0.45

IOU_THRES_NMS    = 0.40
AREA_MIN_PCT     = 0.03               # area minima in % del frame
AREA_MAX_PCT     = 0.30               # area massima in % del frame

# Stabilità
REQ_STABLE_FRAMES = 4                 # n° di frame consecutivi in cui il target resta consistente
STABLE_IOU        = 0.40              # IoU minima tra frame consecutivi per contare stabilità

# Countdown finale
COUNTDOWN_SEC   = 3

# GUI
WINDOW_NAME     = "YOLO — Tools"
FONT            = cv2.FONT_HERSHEY_SIMPLEX

# ====================================
def bbox_area_pct(b, W, H):
    return max(0, b[2]-b[0]) * max(0, b[3]-b[1]) / float(W*H + 1e-9)

def iou_xyxy(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    areaB = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / (areaA + areaB - inter + 1e-9)

def it_name(nm):
    return {

        "drill":"trapano",
        "hammer":"martello",
        "pliers":"pinza",
        "screwdriver":"cacciavite",
        "wrench":"chiave",
       

    }.get(nm,nm)

# ================== MODELLO ==================
model = YOLO(WEIGHTS_PATH)
model.overrides["conf"] = 0.10
model.overrides["iou"]  = IOU_THRES_NMS

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

# Mappa classe selezionata → id 
model_names = getattr(model, "names", None)
if model_names is None:
    model_names = getattr(getattr(model, "model", None), "names", None)
if isinstance(model_names, dict):
    selected_id = [k for k, v in model_names.items() if v == selected][0]
else:
    selected_id = [i for i, v in enumerate(model_names) if v == selected][0]
model.overrides["classes"] = [selected_id]  # << filtro fisso per questa run

# ================== CAMERA ==================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam non disponibile")
time.sleep(0.1)

# Stati
stable_count      = 0
last_box          = None
countdown_start   = None
final_coords      = None
final_frame_bgr   = None
raw_frame_bgr     = None  # frame senza overlay per ritaglio

#  (BGR)
palette_orange    = (0, 165, 255)
palette_red       = (0, 0, 255)
text_blue         = (255, 0, 0)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    H, W = frame.shape[:2]

    # ---------------- YOLO  ----------------
    r = model.predict(source=frame, device=DEVICE, verbose=False)[0]
    boxes = r.boxes.xyxy.cpu().numpy() if r.boxes.xyxy is not None else np.zeros((0,4))
    confs = r.boxes.conf.cpu().numpy()  if r.boxes.conf is not None else np.zeros((0,))
    clses = r.boxes.cls.cpu().numpy()   if r.boxes.cls  is not None else np.zeros((0,))
    names = r.names

    # Raccogli SOLO detezioni della classe selezionata, filtrate per area e conf
    dets = []  # (box, conf)
  
    # Soglia in base al tool selezionato
    if selected == "hammer":
        thr = CONF_HAMMER
    elif selected == "screwdriver":
        thr = CONF_SCREWDRIVER
    elif selected == "drill":
        thr = CONF_DRILL
    elif selected == "pliers":
        thr = CONF_PLIERS
    elif selected == "wrench":
        thr = CONF_WRENCH
    else:
        thr = 0.45  # default

    
    for b, c, k in zip(boxes, confs, clses):
        nm = names[int(k)]
        if nm != selected:
            continue
        ap = bbox_area_pct(b, W, H)
        if ap < AREA_MIN_PCT or ap > AREA_MAX_PCT:
            continue
        if c >= thr:
            dets.append(([float(b[0]), float(b[1]), float(b[2]), float(b[3])], float(c)))

    # Scegli la migliore detezione (massima conf)
    cand = None  # (box, conf)
    for bx, cf in dets:
        if cand is None or cf > cand[1]:
            cand = (bx, cf)

    # Non in countdown
    if countdown_start is None:
        if cand is not None:
            x1,y1,x2,y2 = map(int, cand[0])
            cv2.rectangle(frame, (x1,y1), (x2,y2), palette_orange, 2)
            cv2.putText(frame, f"{cand[1]:.2f}", (x1+90, max(18,y1-8)), FONT, 0.60, palette_orange, 2)

            if last_box is None:
                stable_count = 1
            else:
                iou = iou_xyxy(last_box, cand[0])
                stable_count = stable_count + 1 if iou >= STABLE_IOU else 1
            last_box = cand[0][:]

            cv2.putText(frame, f"Stabilita': {stable_count}/{REQ_STABLE_FRAMES}",
                        (20, H-20), FONT, 0.7, text_blue, 2)
            print(f"Forse ho trovato {it_name(selected)}...")

            if stable_count >= REQ_STABLE_FRAMES:
                countdown_start = time.time()
                print(f"Trovato candidato stabile: inizio countdown {COUNTDOWN_SEC}s.")
        else:
            stable_count = 0
            last_box = None
            cv2.putText(frame, f"Cerco {it_name(selected)} ",
                        (20, 30), FONT, 0.7, text_blue, 2)

    else:
        # COUNTDOWN in corso
        lock = None
        best_iou = -1.0
        for bx, cf in dets:
            iou = iou_xyxy(last_box, bx)
            if iou > best_iou:
                best_iou = iou; lock = (bx, cf)

        if lock is None or best_iou < STABLE_IOU:
            countdown_start = None
            stable_count = 0
            last_box = None
            cv2.putText(frame, "Countdown annullato: target perso, ricerco",
                        (20, 30), FONT, 0.7, palette_red, 2)
            print("Target perso, annullato countdown e ricerco.")
        else:
            # Frame "pulito" prima di disegnare overlay
            raw_frame_bgr = frame.copy()

            # Box e misure (con clamp ai bordi)
            x1, y1, x2, y2 = map(int, lock[0])
            x1 = max(0, min(x1, W-1))
            y1 = max(0, min(y1, H-1))
            x2 = max(0, min(x2, W-1))
            y2 = max(0, min(y2, H-1))
            w  = max(1, x2 - x1)
            h  = max(1, y2 - y1)
            # Centro
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Disegno box rosso + countdown
            cv2.rectangle(frame, (x1,y1), (x2,y2), palette_red, 3)
            cv2.putText(frame, f"{lock[1]:.2f}", (x1+90, max(18,y1-8)), FONT, 0.60, palette_red, 2)

            elapsed = time.time() - countdown_start
            remaining = max(0, COUNTDOWN_SEC - int(np.floor(elapsed)))
            cv2.putText(frame, f"COUNTDOWN: {remaining}", (20, 60), FONT, 0.9, palette_red, 2)

            if elapsed >= COUNTDOWN_SEC:
                print(f"TROVATO {it_name(selected)}.")
                final_coords = (cx, cy)

                # ----- Immagini da salvare -----
                # 1) Overlay intero
                final_frame_bgr = frame.copy()
                cv2.rectangle(final_frame_bgr, (x1,y1), (x2,y2), palette_red, 3)
                cv2.putText(final_frame_bgr, f"centro: ({cx:.1f}, {cy:.1f})", (20, 80), FONT, 0.7, palette_red, 2)
                cv2.putText(final_frame_bgr, f"w={w}, h={h}", (20, 110), FONT, 0.7, palette_red, 2)
                cv2.circle(final_frame_bgr, (int(cx), int(cy)), 5, (0, 0, 255), -1)

                # 2) Frame pulito (senza overlay)
                raw_to_save = raw_frame_bgr.copy()

                # 3) Ritaglio (crop) dal frame pulito, usando la box clamped
                crop = raw_to_save[y1+10:y1+h+10, x1+10:x1+w+10]
                if crop.size == 0:
                    print("Attenzione: crop vuoto (box fuori dai limiti?)")

                out_annot = "screenshot_finale.png"   # intero con overlay
                out_raw   = "frame_raw.png"           # intero pulito
                out_crop  = "box.png"                # SOLO rettangolo ritagliato

            
                if crop.size != 0:
                    cv2.imwrite(out_crop,  crop)

                # ----- Riepilogo + TXT -----
                print("\n=== RISULTATO ===")
                print(f"Tool: {it_name(selected)}  |  Conf = {lock[1]:.2f}")
                print(f"Centro (cx, cy): ({cx:.1f}, {cy:.1f})")
                print(f"Box: x1={x1}, y1={y1}, width={w}, height={h}")
               
                
                print(f"Crop:                {out_crop if crop.size != 0 else 'N/D'}")

                with open("coordinate.txt", "w") as f:
                    f.write(f"tool={it_name(selected)}\n")
                    f.write(f"conf={lock[1]:.4f}\n")
                    f.write(f"cx={cx:.1f}\ncy={cy:.1f}\n")
                    f.write(f"x1={x1}\ny1={y1}\nwidth={w}\nheight={h}\n")
            
                    f.write(f"box_frame={'box.png' if crop.size != 0 else 'NaN'}\n")
                print("File salvato: coordinate.txt")

                # Mostra risultato overlay
                cap.release()
                cv2.imshow(WINDOW_NAME, final_frame_bgr)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                break

    # Streaming
    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break

# Cleanup
if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()