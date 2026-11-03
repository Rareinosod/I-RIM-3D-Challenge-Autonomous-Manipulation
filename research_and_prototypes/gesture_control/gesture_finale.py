"""
associa ai tools:
- Cacciavite 1 → solo pollice esteso
- Cacciavite 2 → pollice e indice estesi
- Cacciavite 3 → pollice, indice e medio estesi
- Martello → pugno chiuso
- Chiave Inglese → indice e mignolo estesi
- Pinza → pollice e mignolo estesi

Un gesto viene confermato solo se rimane stabile per 10 frame consecutivi.
Durante il conteggio il testo è rosso, al termine diventa verde.

Al momento della conferma vengono salvati:
[ nome strumento , mano (Left/Right) , coordinate palmo (x,y) ]

Alla chiusura del programma, la lista del tool confermato viene stampato.

ATTENZIONE!!!
LE MANI DEL UTENDE DEVONO ESSERE POSIZIONATE IN MODO TALE CHE I POLLICI PUNTINO SEMPRE VERSO SINISTRA
 OSSIA LA MANO DETSRAHA IL PALO VERSO LA TELECAMERA E LA MANO SINISTRA HA IL PALMO VERSO SE STESSI 

Regole:
- Conferma solo se il gesto appartiene alle 6 classi e resta stabile per 10 frame consecutivi.
- Stampa SUBITO la prima selezione (senza chiudere la camera).
- Dopo la prima selezione, blocca ulteriori conferme finché non riavvii lo script.

per testare codice ho messo come valore frame attesa 20 ( circa 5 secondi) ma puo essere variato per attendre di piu o di meno 
"""

import time
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

VALID_GESTURES = {
    "Cacciavite 1",
    "Cacciavite 2",
    "Cacciavite 3",
    "Martello",
    "Chiave Inglese",
    "Pinza",
}

def fingers_up(landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    pip_ids  = [3, 6, 10, 14, 18]
    fingers = [False] * 5

    # Pollice: confronto in x (pollici verso sinistra come da assunzione)
    if landmarks[tips_ids[0]][0] < landmarks[2][0]:
        fingers[0] = True

    # Altre dita: tip sopra PIP (in y)
    for i in range(1, 5):
        if landmarks[tips_ids[i]][1] < landmarks[pip_ids[i]][1]:
            fingers[i] = True

    return fingers

def classify_gesture(fingers):
    thumb, index, middle, ring, pinky = fingers

    # Cacciavite 1 → solo pollice
    if thumb and not any([index, middle, ring, pinky]):
        return "Cacciavite 1"

    # Cacciavite 2 → pollice + indice
    if thumb and index and not any([middle, ring, pinky]):
        return "Cacciavite 2"

    # Cacciavite 3 → pollice + indice + medio
    if thumb and index and middle and not any([ring, pinky]):
        return "Cacciavite 3"

    # Martello → pugno chiuso
    if not any([thumb, index, middle, ring, pinky]):
        return "Martello"

    # Chiave Inglese → indice + mignolo
    if index and pinky and not any([thumb, middle, ring]):
        return "Chiave Inglese"

    # Pinza → pollice + mignolo
    if thumb and pinky and not any([index, middle, ring]):
        return "Pinza"

    return None


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Errore nell'aprire la fotocamera.")
    else:
        print("Fotocamera aperta correttamente.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_threshold = 20  # stabilità richiesta

    selected_items = []          # conterrà al massimo 1 elemento
    selection_locked = False     # blocca dopo la prima conferma

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        prev_time = 0.0

        # Stato per la stabilità
        last_gesture = None
        last_hand = None
        countdown = 0

        # Per disegnare fisso cosa è stato selezionato
        locked_display_text = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            any_hand_detected = False

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    any_hand_detected = True

                    # Disegno landmarks
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=1, circle_radius=1),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )

                    # Centro polso
                    wrist_x = int(hand_landmarks.landmark[0].x * w)
                    wrist_y = int(hand_landmarks.landmark[0].y * h)

                    # Se già bloccato, mostra solo il selezionato e ignora tutto il resto
                    if selection_locked:
                        if locked_display_text:
                            cv2.putText(frame, locked_display_text, (20, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        continue

                    # Altrimenti, continuiamo con il riconoscimento
                    lm = [(l.x, l.y) for l in hand_landmarks.landmark]
                    fingers = fingers_up(lm)
                    gesture = classify_gesture(fingers)   # None se non valida
                    hand_label = handedness.classification[0].label  # "Left" / "Right"
                    text = f"{hand_label}: {gesture}"

                    # Gestione stabilità SOLO per gesti validi
                    if gesture in VALID_GESTURES:
                        if gesture == last_gesture and hand_label == last_hand:
                            countdown += 1
                        else:
                            countdown = 1
                            last_gesture = gesture
                            last_hand = hand_label

                        # Disegno contatore/stato
                        color = (0, 0, 255) if countdown < frame_threshold else (0, 255, 0)
                        cv2.putText(frame, f"{hand_label}: {gesture} ({countdown}/{frame_threshold})",
                                    (max(wrist_x - 150, 10), max(wrist_y - 30, 30)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                        # Conferma: PRIMA e UNICA
                        if countdown >= frame_threshold and not selection_locked:
                            item = [gesture, hand_label, (wrist_x, wrist_y)]
                            selected_items.append(item)
                            selection_locked = True
                            locked_display_text = f"SELEZIONATO: {gesture} ({hand_label}) {item[2]}"
                            # Stampa IMMEDIATA (senza chiudere)
                            print("Oggetti selezionati:", selected_items)

                    else:
                        # Gesto non valido → reset conteggio (solo se non bloccato)
                        countdown = 0
                        last_gesture = None
                        last_hand = None
                        cv2.putText(frame, "Gesture non riconosciuta",
                                    (max(wrist_x - 150, 10), max(wrist_y - 30, 30)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Se nessuna mano → reset “soft” se non bloccato
            if not any_hand_detected and not selection_locked:
                countdown = 0
                last_gesture = None
                last_hand = None

            # Banner fisso quando bloccato
            if selection_locked and locked_display_text:
                cv2.putText(frame, locked_display_text, (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0.0
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Mostra
            cv2.imshow("Gesture Recognition - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()