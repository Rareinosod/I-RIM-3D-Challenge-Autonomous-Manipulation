import time
import cv2
import numpy as np
from ultralytics import YOLO

# ================== CONFIG ==================
WEIGHTS_PATH   = "best.pt"
DEVICE         = "cpu"               # "cpu" oppure 0 per GPU
CLASSES        = {"hammer", "screwdriver"}  # martello, cacciavite

# Soglie & filtri
CONF_HAMMER    = 0.65                # conf minima per considerare valido il martello
CONF_SCREWDRIVER = 0.50              # conf minima per considerare valido il cacciavite
CONF_DRAW_MIN  = 0.10                # conf minima per mostrare box blu
IOU_THRES_NMS  = 0.40
AREA_MIN_PCT   = 0.03                #  area frame (min)
AREA_MAX_PCT   = 0.30                # area frame (max)

# Stabilità (richiesta dall'utente)
REQ_STABLE_FRAMES = 5               # n° di frame consecutivi in cui il target resta consistente
STABLE_IOU        = 0.40             # IoU minima tra frame consecutivi per contare stabilità

# Countdown finale
COUNTDOWN_SEC  = 1

# GUI
WINDOW_NAME    = "YOLO — Tools"
FONT           = cv2.FONT_HERSHEY_SIMPLEX

# ================== UTILS ==================
def bbox_area_pct(b, W, H):
    """ Calcola l'area della bounding box in percentuale rispetto al frame """
    return max(0, b[2]-b[0]) * max(0, b[3]-b[1]) / float(W*H + 1e-9)

def iou_xyxy(a, b):
    """ Calcola l'Intersection over Union (IoU) tra due box """
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    areaB = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / (areaA + areaB - inter + 1e-9)

def it_name(nm):  # etichetta italiana
    return "martello" if nm=="hammer" else ("cacciavite" if nm=="screwdriver" else nm)

# ================== MODEL ==================
model = YOLO(WEIGHTS_PATH) #posso mettere una classe specifica cosi codice piu leggere 
model.overrides["conf"] = 0.10  # lasciamo bassa per vedere box blu; filtriamo noi dopo
model.overrides["iou"]  = IOU_THRES_NMS

# ================== INPUT SCELTA ==================
print("Seleziona il tool da cercare:")
print("1 - Martello")
print("2 - Cacciavite")
selected = None
while True:
    choice = input("Seleziona 1 (Martello) o 2 (Cacciavite): ").strip()
    if choice == "1":
        selected = "hammer"
        print("Hai selezionato: Martello")
        break
    elif choice == "2":
        selected = "screwdriver"
        print("Hai selezionato: Cacciavite")
        break
    else:
        print("Strumento non riconosciuto. Riprova.")

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

palette_blue      = (255, 0, 0)
palette_orange    = (0, 165, 255)
palette_red       = (0, 0, 255)
text_white        = (255, 0, 0)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    H, W = frame.shape[:2]

    # ---------------- YOLO ----------------
    r = model.predict(source=frame, device=DEVICE, verbose=False)[0]
    boxes = r.boxes.xyxy.cpu().numpy() if r.boxes.xyxy is not None else np.zeros((0,4))
    confs = r.boxes.conf.cpu().numpy()  if r.boxes.conf is not None else np.zeros((0,))
    clses = r.boxes.cls.cpu().numpy()   if r.boxes.cls  is not None else np.zeros((0,))
    names = r.names

    # Raccogliamo detezioni filtrate per area e conf min di draw
    dets = []  # (name, box, conf)
    for b, c, k in zip(boxes, confs, clses):
        nm = names[int(k)]
        if nm not in CLASSES:
            continue
        ap = bbox_area_pct(b, W, H)
        if ap < AREA_MIN_PCT or ap > AREA_MAX_PCT:
            continue
        # Se il nome è martello e la confidenza è sopra la soglia del martello
        if nm == "hammer" and c >= CONF_HAMMER:
            dets.append((nm, [float(b[0]), float(b[1]), float(b[2]), float(b[3])], float(c)))
        # Se il nome è cacciavite e la confidenza è sopra la soglia del cacciavite
        elif nm == "screwdriver" and c >= CONF_SCREWDRIVER:
            dets.append((nm, [float(b[0]), float(b[1]), float(b[2]), float(b[3])], float(c)))
        # Se la confidenza è inferiore, scarta l'oggetto

    # Disegno TUTTE le box in blu + conf
    for nm, bx, cf in dets:
        x1,y1,x2,y2 = map(int, bx)
        cv2.rectangle(frame, (x1,y1), (x2,y2), palette_blue, 2)
        cv2.putText(frame, f"{it_name(nm)} ", (x1, max(18,y1-8)), FONT, 0.55, palette_blue, 2)

    # ---- cerca la migliore detezione della classe selezionata (massima conf) ----
    cand = None  # (box, conf)
    for nm, bx, cf in dets:
        if nm == selected and cf >= (CONF_HAMMER if selected == "hammer" else CONF_SCREWDRIVER):
            if cand is None or cf > cand[1]:
                cand = (bx, cf)

    # Se non siamo in countdown:
    if countdown_start is None:
        if cand is not None:
            # evidenzia in arancione il candidato attuale
            x1,y1,x2,y2 = map(int, cand[0])
            cv2.rectangle(frame, (x1,y1), (x2,y2), palette_orange, 2)
            cv2.putText(frame, f"{cand[1]:.2f}",
                        (x1+90, max(18,y1-8)), FONT, 0.60, palette_orange, 2)

            # stabilità su N frame con IoU
            if last_box is None:
                stable_count = 1
            else:
                iou = iou_xyxy(last_box, cand[0])
                stable_count = stable_count + 1 if iou >= STABLE_IOU else 1
            last_box = cand[0][:]

            # feedback su gui e terminale
            cv2.putText(frame, f"Stabilita': {stable_count}/{REQ_STABLE_FRAMES}",
                        (20, H-20), FONT, 0.7, text_white, 2)

            # Terminale feedback per l'utente
            print(f"Forse ho trovato {it_name(selected)}...")

            # se stabile abbastanza → parte il countdown
            if stable_count >= REQ_STABLE_FRAMES:
                countdown_start = time.time()
                print(f"trovato,inizio a salvare le coordinate {it_name(selected)} !")
        else:
            # target non valido/assente → reset stabilita'
            stable_count = 0
            last_box = None
            cv2.putText(frame, f"Cerco {it_name(selected)} ",
                        (20, 30), FONT, 0.7, text_white, 2)

    else:
        # COUNTDOWN in corso: il box diventa ROSSO. Verifichiamo che il target resti presente.
        lock = None
        best_iou = -1.0
        for nm, bx, cf in dets:
            if nm != selected or cf < (CONF_HAMMER if selected == "hammer" else CONF_SCREWDRIVER):
                continue
            iou = iou_xyxy(last_box, bx)
            if iou > best_iou:
                best_iou = iou; lock = (bx, cf)

        # se target perso → annulla countdown e torna a stabilizzare
        if lock is None or best_iou < STABLE_IOU:
            countdown_start = None
            stable_count = 0
            last_box = None
            cv2.putText(frame, "Countdown annullato: target perso ricerco", (20, 30), FONT, 0.7, palette_red, 2)
            print("Target perso, annullato countdown e ricerco.")

        else:
            # disegna rosso + countdown visibile
            x1,y1,x2,y2 = map(int, lock[0])
            cv2.rectangle(frame, (x1,y1), (x2,y2), palette_red, 3)
            cv2.putText(frame, f" {lock[1]:.2f}",
                        (x1+90, max(18,y1-8)), FONT, 0.60, palette_red, 2)

            elapsed = time.time() - countdown_start
            remaining = max(0, COUNTDOWN_SEC - int(np.floor(elapsed)))
            cv2.putText(frame, f"COUNTDOWN: {remaining}", (W-1240, 40), FONT, 0.9, palette_red, 2)

            # Messaggio countdown finale
            if elapsed >= COUNTDOWN_SEC:
                print(f"TROVATO {it_name(selected)}.")
                # FINISH: salva frame + coordinate e chiudi camera
                x1, y1, x2, y2 = map(int, lock[0])  
                # Calcola il centro del box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                final_coords = (cx, cy)
                # Fisso un frame finale con overlay (box rosso + testo)
                final_frame_bgr = frame.copy()
                cv2.rectangle(final_frame_bgr, (x1,y1), (x2,y2), palette_red, 3)
                cv2.putText(final_frame_bgr, f"centro: {final_coords}", (20, 70), FONT, 0.7, palette_red, 2)
                # Disegnare il punto rosso al centro del box
                cv2.circle(final_frame_bgr, (int(cx), int(cy)), 5, (0, 0, 255), -1)  # 5 è il raggio del cerchio, (0, 0, 255) è il rosso
                

                # screenshot
                out_name = "screenshot_finale.png"
                cv2.imwrite(out_name, final_frame_bgr)
                print(f"\n=== RISULTATO ===")
                print(f"Tool: {it_name(selected)}  |  Conf = {lock[1]:.2f}")
                print(f"Coordinate del centro della box: {final_coords}")
                print(f"Screenshot salvato: {out_name}")

                # Chiudo la camera e poi mostro l'ultimo frame finché premi un tasto
                cap.release()
                cv2.imshow(WINDOW_NAME, final_frame_bgr)
                cv2.waitKey(0)  # resta visibile per debug finché premi un tasto
                cv2.destroyAllWindows()
                break  # esco dal while principale

    # mostra streaming
    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break

# cleanup se esci senza finale
if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()