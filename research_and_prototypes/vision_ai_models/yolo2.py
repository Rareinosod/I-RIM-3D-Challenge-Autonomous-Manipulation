#!/usr/bin/env python3
import cv2
import numpy as np
from ultralytics import YOLO
import os
import logging
import glob # Importato per caricare i file

# Configura il logging di base
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OfflineObjectProcessor:
    def __init__(self, data_folder, model_path):
        """
        Inizializza il processore offline.
        Logica di classificazione cacciaviti basata su
        CONFRONTO ISTOGRAMMI (cv2.compareHist) con impronte hardcoded.
        Rimossa la detection dei box colorati.
        """
        logging.info("=== Avvio del Processore Oggetti Offline (YOLO + Colore Cacciavite) ===")

        # === PARAMETRI ===
        self.folder = data_folder
        self.model_path = model_path
        
        # Cartella per i risultati visualizzati
        self.output_folder = os.path.join(self.folder, "processing_results")
        os.makedirs(self.output_folder, exist_ok=True)
        logging.info(f"I risultati saranno salvati in: {self.output_folder}")

        if not os.path.isdir(self.folder):
            logging.error(f"La cartella specificata non esiste: {self.folder}")
            raise FileNotFoundError(f"La cartella specificata non esiste: {self.folder}")
        if not os.path.isfile(self.model_path):
            logging.error(f"Il modello YOLO non è stato trovato: {self.model_path}")
            raise FileNotFoundError(f"Il modello YOLO non è stato trovato: {self.model_path}")

       
        # === YOLO MODEL ===
        self.model = YOLO(self.model_path)
        self.model.to("cpu")
        logging.info("Modello YOLO caricato su CPU.")
            
        # Mappa delle classi
        self.CLASS_MAP = {
            0: "pliers",
            1: "screwdriver",
            2: "wrench"
        }
        self.CLASS = list(self.CLASS_MAP.keys()) 
        logging.info(f"Cerco le classi YOLO con ID: {self.CLASS}")

        # === IMPRONTE ISTOGRAMMI (HARDCODED) ===
        # Dati generati dallo script 'genera_impronte.py'
        
        # !!! INCOLLA QUI LE TUE DUE RIGHE !!!
        self.hist_red = np.array([[3.59717548e-01],
       [8.61251205e-02],
       [3.19862366e-02],
       [1.45226195e-02],
       [1.61063764e-02],
       [1.84460524e-02],
       [1.13565568e-02],
       [1.00655705e-02],
       [1.75266098e-02],
       [1.52857527e-02],
       [1.44072054e-02],
       [2.44969334e-02],
       [1.93320978e-02],
       [2.36842036e-02],
       [2.30147261e-02],
       [3.50023806e-02],
       [5.32849096e-02],
       [1.26114905e-01],
       [2.82101780e-01],
       [3.74681532e-01],
       [1.00000000e+00],
       [7.91394770e-01],
       [8.39276835e-02],
       [1.82287954e-03],
       [2.12588304e-04],
       [2.30269598e-05],
       [1.79471135e-06],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [3.33060816e-05],
       [8.32652040e-06],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [1.04081497e-04],
       [0.00000000e+00],
       [2.08163001e-05],
       [7.91019411e-05],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [2.49795612e-05],
       [0.00000000e+00],
       [0.00000000e+00],
       [5.41223817e-05],
       [0.00000000e+00],
       [7.89672995e-05],
       [0.00000000e+00],
       [6.17515354e-04],
       [1.78739156e-05],
       [7.17884541e-06],
       [1.21860905e-03],
       [3.58942270e-06],
       [1.66101119e-04],
       [0.00000000e+00],
       [5.56360523e-04],
       [5.52771089e-04],
       [3.58942270e-06],
       [0.00000000e+00],
       [4.16443887e-04],
       [1.55731555e-04],
       [0.00000000e+00],
       [8.39443645e-04],
       [4.51476924e-04],
       [7.34617133e-05],
       [1.46435760e-03],
       [4.19774180e-04],
       [1.15929614e-03],
       [2.02945434e-03],
       [3.39847337e-03],
       [7.83365965e-03],
       [1.56477168e-02],
       [2.50381441e-03],
       [1.80913415e-02],
       [6.64931443e-03],
       [1.11339055e-02],
       [9.47791804e-03],
       [2.82905996e-02],
       [6.98897541e-02],
       [1.47220418e-01],
       [2.49424249e-01],
       [4.18676168e-01],
       [3.80701900e-01]], dtype=np.float32)
        

        self.hist_black = np.array([[1.72284269e-03],
       [4.49789135e-04],
       [7.10190565e-04],
       [9.34091338e-04],
       [8.90756841e-04],
       [3.37982411e-03],
       [1.91890146e-03],
       [1.61531521e-03],
       [6.01269864e-03],
       [7.74212461e-03],
       [2.39448156e-02],
       [4.21564430e-02],
       [7.79732242e-02],
       [9.63925868e-02],
       [1.69194058e-01],
       [5.99592142e-02],
       [3.96298850e-03],
       [4.89167054e-04],
       [1.43876532e-04],
       [9.65222644e-05],
       [0.00000000e+00],
       [6.67264467e-05],
       [0.00000000e+00],
       [0.00000000e+00],
       [2.22421477e-05],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [8.89685907e-05],
       [6.67264467e-05],
       [1.33452893e-04],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [2.22421477e-05],
       [0.00000000e+00],
       [4.44842954e-05],
       [4.44842954e-05],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [2.22421477e-05],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [2.22421477e-05],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [2.22421477e-05],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [4.44842954e-05],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [0.00000000e+00],
       [1.11210735e-04],
       [0.00000000e+00],
       [0.00000000e+00],
       [9.00891973e-05],
       [0.00000000e+00],
       [0.00000000e+00],
       [4.44842954e-05],
       [0.00000000e+00],
       [1.48517924e-04],
       [1.15763175e-03],
       [0.00000000e+00],
       [9.10258736e-04],
       [0.00000000e+00],
       [0.00000000e+00],
       [1.12813734e-03],
       [7.26758386e-04],
       [1.29575834e-01],
       [1.61777381e-02],
       [5.33961728e-02],
       [4.20671076e-01],
       [2.21253559e-01],
       [8.72246504e-01],
       [9.99999940e-01],
       [9.13468897e-01],
       [9.57218468e-01],
       [3.38383436e-01],
       [7.92085901e-02],
       [9.89915133e-02],
       [3.11564431e-02],
       [7.79818185e-03],
       [1.16523053e-03],
       [9.25810018e-04],
       [6.09872746e-04],
       [7.41266925e-03],
       [0.00000000e+00],
       [1.31869153e-03],
       [1.67504360e-04],
       [9.76564246e-04],
       [2.50865798e-03],
       [1.12580610e-02],
       [1.47482991e-04],
       [6.19913742e-04],
       [3.86447384e-04],
       [2.45975360e-04],
       [0.00000000e+00],
       [1.36503848e-04],
       [4.29376902e-04],
       [2.33627525e-05],
       [3.44328553e-04],
       [1.04303770e-04],
       [0.00000000e+00],
       [5.58789179e-04],
       [0.00000000e+00],
       [0.00000000e+00],
       [1.48935331e-04],
       [2.99946696e-04],
       [0.00000000e+00],
       [8.65633483e-05],
       [8.58590574e-05],
       [1.67486476e-04],
       [7.96756867e-05],
       [1.16813762e-05],
       [0.00000000e+00],
       [2.33217725e-04],
       [0.00000000e+00],
       [1.16813762e-05],
       [3.09097464e-04],
       [0.00000000e+00],
       [1.32460234e-04],
       [3.55304626e-04],
       [7.67710226e-05],
       [1.61819335e-04],
       [1.00133766e-04],
       [8.22679867e-05],
       [5.00668830e-05],
       [1.16813762e-05],
       [2.48302124e-04],
       [4.77991474e-04],
       [0.00000000e+00],
       [1.19326520e-04],
       [3.50441260e-05],
       [0.00000000e+00],
       [3.77606979e-04],
       [9.98132280e-04],
       [1.16813762e-05],
       [0.00000000e+00],
       [4.67255049e-05],
       [5.91962191e-04],
       [3.66441818e-04],
       [2.33627525e-05],
       [9.94174858e-04],
       [2.83925183e-04],
       [1.07582469e-04]], dtype=np.float32)

        # Controllo che le matrici siano state incollate
        if self.hist_red is None or self.hist_black is None:
            logging.error("ERRORE: Devi incollare le matrici self.hist_red e self.hist_black nell' __init__.")
            raise ValueError("Impronte istogrammi non caricate.")
        # Controllo (opzionale ma utile) che abbiano la forma giusta
        elif self.hist_red.shape != (180, 1) or self.hist_black.shape != (180, 1):
             logging.warning(f"ATTENZIONE: Le dimensioni degli istogrammi potrebbero essere errate.")
             logging.warning(f"Dimensione hist_red: {self.hist_red.shape}, attesa: (180, 1)")
             logging.warning(f"Dimensione hist_black: {self.hist_black.shape}, attesa: (180, 1)")
             # Non blocchiamo l'esecuzione, ma avvisiamo
        else:
            logging.info("Impronte istogrammi 'red' e 'black' caricate.")

    # ---------------------- FILE HANDLING ----------------------

    def load_data(self, folder):
        """Carica tutti i file immagine nella cartella."""
        patterns = ['*.png', '*.jpg', '*.jpeg', '*.bmp','*.DNG'] # Aggiunto DNG se serve
        image_files = []
        for pattern in patterns:
            image_files.extend(glob.glob(os.path.join(folder, pattern)))
        
        logging.info(f"Trovati {len(image_files)} file immagine da processare.")
        # Ordina i file per nome per un'elaborazione consistente (opzionale)
        image_files.sort() 
        return image_files

    # ---------------------- BOX DETECTION (Rimossa) ----------------------
    # def detect_colored_boxes(self, image): ... (Rimosso)

    # ---------------------- SEGMENTAZIONE ----------------------
    def mask_segmentation(self, cropped):
        """Segmentazione tramite Otsu e componenti connesse."""
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        # Usiamo THRESH_OTSU che trova la soglia automaticamente
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Operazioni morfologiche per pulire la maschera
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        m = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
        # Trova il componente connesso più grande (l'oggetto principale)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if num_labels <= 1: # Solo sfondo trovato
            return None # Restituisce None se la segmentazione fallisce
        else:
            # Ignora l'etichetta 0 (sfondo)
            areas = stats[1:, cv2.CC_STAT_AREA]
            best_label = 1 + np.argmax(areas) # Aggiunge 1 perché abbiamo ignorato lo sfondo
            mask = (labels == best_label).astype(np.uint8) * 255
        return mask

    # ---------------------- PROCESSO 2D ----------------------
    def process_data(self):
        """
        Funzione principale: carica immagini, esegue YOLO, classifica
        colore cacciaviti e salva immagini annotate.
        """
        
        self.image_files = self.load_data(self.folder)

        if not self.image_files:
            logging.error("Nessun file immagine trovato. Interruzione.")
            return

        for i, img_file in enumerate(self.image_files):
            logging.info(f"Processing frame {i+1}/{len(self.image_files)}: {img_file}")
            
            frame = cv2.imread(img_file)
            if frame is None:
                logging.warning(f"Impossibile leggere il file immagine: {img_file}")
                continue
            
            output_image = frame.copy()
            
            # === TOOL DETECTION (YOLO) ===
            results = self.model.predict(frame, conf=0.25, iou=0.35, classes=self.CLASS, agnostic_nms=True)

            if not results or len(results[0].boxes) == 0:
                logging.warning(f"Nessun oggetto YOLO trovato in {img_file}")
                # Salva comunque l'immagine originale se non trova nulla? Opzionale.
                # output_path = os.path.join(self.output_folder, os.path.basename(img_file))
                # cv2.imwrite(output_path, output_image)
                continue # Passa all'immagine successiva

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0].item())
                cls_name = self.get_class_name(cls_id)

                if "unknown" in cls_name:
                    logging.warning(f"Trovata classe sconosciuta con ID: {cls_id} in {img_file}")
                    continue # Ignora classi sconosciute

                # Crop leggermente allargato per segmentazione
                x1_c = max(0, x1 - 10)
                y1_c = max(0, y1 - 10)
                x2_c = min(frame.shape[1], x2 + 10)
                y2_c = min(frame.shape[0], y2 + 10)
                
                cropped = frame[y1_c:y2_c, x1_c:x2_c]
                if cropped.size == 0:
                    logging.warning(f"Crop nullo per {cls_name} in {img_file}")
                    continue
                    
                mask_2D = self.mask_segmentation(cropped)
                if mask_2D is None:
                    # Se la segmentazione fallisce, usa l'intero crop come maschera
                    logging.warning(f"Segmentazione fallita per {cls_name} in {img_file}, uso l'intero crop.")
                    mask_2D = np.ones(cropped.shape[:2], dtype=np.uint8) * 255
                
                # Applica maschera (utile per istogramma)
                cropped_segmented = cv2.bitwise_and(cropped, cropped, mask=mask_2D)


                # === LOGICA COLORE CACCIAVITE (CONFRONTO ISTOGRAMMI) ===
                if "screwdriver" in cls_name:
                    hsv = cv2.cvtColor(cropped_segmented, cv2.COLOR_BGR2HSV)
                    h, s, v = cv2.split(hsv)

                    # Filtri per istogramma (usa i valori che funzionavano)
                    s_min = 45 
                    v_min = 50
                    mask_hist = (s > s_min) & (v > v_min) & (mask_2D > 0) # Usa mask_2D originale

                    # Calcola istogramma HUE del cacciavite trovato
                    hist_current = cv2.calcHist([hsv], [0], mask_hist.astype(np.uint8), [180], [0, 180])

                    num_valid_pixels = np.sum(mask_hist)

                    if num_valid_pixels < 20: 
                        cls_name = "screwdriver_black" # Default se pochi pixel
                        logging.debug(f"Pochi pixel ({num_valid_pixels}) per istogramma in {img_file}, default a black.")
                    else:
                        cv2.normalize(hist_current, hist_current, 0, 1, cv2.NORM_MINMAX)
                        
                        try:
                            # Confronta usando BHATTACHARYYA (punteggio più BASSO è migliore)
                            score_red = cv2.compareHist(self.hist_red, hist_current, cv2.HISTCMP_BHATTACHARYYA)
                            score_black = cv2.compareHist(self.hist_black, hist_current, cv2.HISTCMP_BHATTACHARYYA)
                            
                            logging.debug(f"Scores per {os.path.basename(img_file)}: Red={score_red:.4f}, Black={score_black:.4f}")

                            if score_red < score_black: 
                                cls_name = "screwdriver_red"
                            else:
                                cls_name = "screwdriver_black"
                        except cv2.error as e:
                            logging.error(f"Errore OpenCV durante compareHist in {img_file}: {e}")
                            cls_name = "screwdriver_black" # Default in caso di errore
                        logging.debug(f"Classificato come: {cls_name}")

                # --- Disegna Risultati YOLO ---
                # Colori diversi per classe per chiarezza
                color = (0, 255, 0) # Verde default
                if "red" in cls_name:
                    color = (0, 0, 255) # Rosso
                elif "black" in cls_name:
                    color = (50, 50, 50) # Grigio scuro
                elif "pliers" in cls_name:
                    color = (255, 0, 0) # Blu
                elif "wrench" in cls_name:
                    color = (0, 255, 255) # Giallo
                
                # Disegna il box originale di YOLO (x1, y1, x2, y2)
                cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
                # Aggiungi etichetta
                cv2.putText(output_image, cls_name, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # === BOX DETECTION (Rimossa) ===
            # colored_boxes = self.detect_colored_boxes(frame) ... (Rimosso)
            # for box_det in colored_boxes: ... (Rimosso)

            # === SALVA IMMAGINE CON ANNOTAZIONI ===
            output_filename = os.path.basename(img_file)
            # Modifica per salvare come JPG (o mantieni PNG se preferisci)
            output_filename_base, _ = os.path.splitext(output_filename)
            output_path = os.path.join(self.output_folder, output_filename_base + ".jpg") 
            try:
                cv2.imwrite(output_path, output_image)
            except Exception as e:
                logging.error(f"Impossibile salvare l'immagine {output_path}: {e}")
                
        logging.info("Elaborazione 2D completata!")
        logging.info(f"Immagini annotate salvate in: {self.output_folder}")
        

    # ---------------------- CLASSI ----------------------
    def get_class_name(self, cls_id):
        """Mappa l'ID della classe YOLO al nome stringa."""
        return self.CLASS_MAP.get(cls_id, f"unknown_{cls_id}")


if __name__ == "__main__":
    
    # --- PERCORSI IMPOSTATI MANUALMENTE ---
    PATH_CARTELLA_IMMAGINI = "/Users/alessandro/Desktop/IRIM_Challenge2025/codici python_visual_recognition/TATA/picture_scene_reale"
    PATH_MODELLO_YOLO = "/Users/alessandro/Desktop/IRIM_Challenge2025/codici python_visual_recognition/TATA/best.pt"
    # --- FINE PERCORSI ---
    
    try:
        # 1. Inizializza il processore
        processor = OfflineObjectProcessor(
            data_folder=PATH_CARTELLA_IMMAGINI, 
            model_path=PATH_MODELLO_YOLO
        )
        
        # 2. Esegui l'elaborazione 2D
        processor.process_data()

    except FileNotFoundError as e:
        logging.error(f"Errore: {e}. Controlla i percorsi e riprova.")
    except ValueError as e: # Cattura l'errore se le matrici non sono incollate
        logging.error(f"Errore di configurazione: {e}")
    except Exception as e:
        logging.error(f"Si è verificato un errore imprevisto: {e}", exc_info=True) # Stampa traceback completo