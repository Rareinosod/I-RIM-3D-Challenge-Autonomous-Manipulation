import cv2
import numpy as np
import os

# --- CONFIGURAZIONE ---

# --- da settare ---
COLOR_RANGES = {
    'verde': {
        # H (Tonalità): 40-90 (ampio, copre molte sfumature di verde)
        # S (Saturazione): min 30 (accetta anche colori poco saturi, quasi grigi)
        # V (Valore): min 30 (accetta colori molto scuri, in ombra)
        'lower1': np.array([40, 30, 30]), 
        'upper1': np.array([90, 255, 255])
    },
    'blu': {
       
        'lower1': np.array([95, 100, 80]), # leggermente più generoso per V e S minimi del blu
        'upper1': np.array([125, 255, 255])
    }
}



# Dizionario per colorare i box di output
OUTPUT_COLORS = {
    'verde': (0, 255, 0),
    'blu': (255, 100, 0)
}



#  l'area del nastro
MIN_RECT_AREA = 1000
MAX_RECT_AREA = 500000 

def find_colored_rectangles(image_path):
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Errore: Immagine non trovata al percorso: {image_path}")
        return None, []

    output_image = image.copy()
    detected_objects = []

    blurred_image = cv2.GaussianBlur(image, (7, 7), 0)
    hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

    for color_name, ranges in COLOR_RANGES.items():
        mask1 = cv2.inRange(hsv_image, ranges['lower1'], ranges['upper1'])
        if 'lower2' in ranges:
            mask2 = cv2.inRange(hsv_image, ranges['lower2'], ranges['upper2'])
            color_mask = cv2.bitwise_or(mask1, mask2)
        else:
            color_mask = mask1

 
        kernel = np.ones((3, 3), np.uint8)
        color_mask = cv2.dilate(color_mask, kernel, iterations=2)
        color_mask = cv2.erode(color_mask, kernel, iterations=1)
    
        contours, hierarchy = cv2.findContours(color_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        if hierarchy is None:
            continue
            
        hierarchy = hierarchy[0]

        for i, contour in enumerate(contours):
            parent = hierarchy[i][3]
            # 1. È un contorno esterno?
            if parent == -1:
                area = cv2.contourArea(contour)
                
                # 2. Ha la dimensione giusta?
                if MIN_RECT_AREA < area < MAX_RECT_AREA:
                    perimeter = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                    
                    # 3. Ha 4 lati?
                    if len(approx) == 4:
                        
                        # 4. Ha un buco (un "figlio")?
                        first_child_index = hierarchy[i][2]
                        if first_child_index != -1:
                            
                            # --- RILEVAMENTO VALIDO! ---
                            # È un bordo rettangolare.
                            
                            (x, y, w, h) = cv2.boundingRect(approx)
                            center_x = x + w // 2
                            center_y = y + h // 2
                            
                            box_color = OUTPUT_COLORS.get(color_name, (0, 255, 0))
                            
                          
                            
                            # Aggiungiamo un cerchio al centro
                            cv2.circle(output_image, (center_x, center_y), 10, (255, 255, 255), -1)
                       
                            
                            detected_objects.append({
                                'color': color_name,
                                'type': 'bordo_riempito',
                                'center_x': center_x,
                                'center_y': center_y,
                                'width': w,
                                'height': h,
                                'area': area
                            })

    return output_image, detected_objects


# ------

if __name__ == "__main__":

   
    INPUT_IMAGE_PATH = "/Users/alessandro/Desktop/IRIM_Challenge2025/codici python_visual_recognition/WhatsApp Image 2025-10-18 at 12.18.45.jpeg"

    
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"ERRORE: Immagine non trovata! Controlla il percorso:")
        print(INPUT_IMAGE_PATH)
    else:
        result_image, detections = find_colored_rectangles(INPUT_IMAGE_PATH)

        if result_image is not None:
            print("\n--- Rilevamento bordi completato ---")
            if detections:
                for item in detections:
                    print(f"Trovato box _{item['color'].upper()} con centro a (x={item['center_x']}, y={item['center_y']})")
            else:
                print("Nessun bordo corrispondente ai criteri è stato trovato.")
             
            
            
          
            
        

            cv2.imshow("Originale  |  Risultato ", result_image)
            
            output_path = "risultato_rilevamento_riempito.png"
            cv2.imwrite(output_path, result_image)
            print(f"Immagine risultato salvata come: {output_path}")
          
            cv2.waitKey(0)
            cv2.destroyAllWindows()