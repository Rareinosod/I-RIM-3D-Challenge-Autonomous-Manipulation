from ultralytics import YOLO
import cv2



# Inizializza la webcam
cap = cv2.VideoCapture(0)
# Initialize a YOLO-World model
model = YOLO("prova.pt")  # or select yolov8m/l-world.pt for different sizes
model.to("cuda")
print(f"Modello caricato su: {model.device}")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define custom classes
    #model.set_classes("screwdriver")#, "wrench", "drill", "saw"]
    # Execute prediction for specified categories on an image 0 = drill, 1 = hammer, 2 = pliers, 
    # 3 = screwdriver, 4 = wrench, 5 = saw
    CLASS = [3]
    results = model.predict(frame, conf=0.25, iou=0.25,classes = CLASS)

    # Ottieni l'immagine con le predizioni disegnate
    annotated_frame = results[0].plot()#(labels=False)

    # Mostra il frame nella finestra
    cv2.imshow("Detection", annotated_frame)

    # Attendi la pressione di un tasto
    key = cv2.waitKey(1) & 0xFF

    # Se il tasto premuto è 'q', esce dal ciclo e chiude la webcam
    if key == ord('q'):  # ord('q') restituisce il codice ASCII per il tasto 'q'
        print("Chiusura della webcam...")
        cap.release()
        cv2.destroyAllWindows()
        exit()
