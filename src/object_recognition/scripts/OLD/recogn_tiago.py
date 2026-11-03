#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import torch
from pathlib import Path

# prova per capire se riconosce cacciavite
class TiagoYoloDetector:
    def __init__(self):
        # Inizializza il nodo ROS
        rospy.init_node("tiago_yolo_detector", anonymous=True)

        # CvBridge per conversione ROS in OpenCV
        self.bridge = CvBridge()

        # Publisher per il nome dell’oggetto rilevato
        self.detection_pub = rospy.Publisher(
            "/yolo_detections", String, queue_size=10
        )

        # Trova la home dell'utente corrente
        self.home = Path.home()
        print(self.home)
        if self.home == Path("/home/pal"):
            self.model_path = self.home / "Documenti" / "GitHub" / "IRIM" / "src" / "object_recognition" / "scripts" / "prova.pt"

        elif self.home == Path("/home/edoardo"):
            self.model_path = self.home / "Documents" / "GitHub" / "IRIM_manipulation_ws" / "src" / "object_recognition" / "scripts" / "prova.pt"


        # Carica il modello YOLO
        self.model = YOLO(self.model_path)
        """# Carica il modello YOLO
        self.model = YOLO("/home/edoardo/Documents/GitHub/IRIM_manipulation_ws/src/object_recognition/scripts/prova.pt") # edo
        self.model = YOLO("/home/pal/Documents/GitHub/IRIM/src/object_recognition/scripts/prova.pt") # rita"""

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        rospy.loginfo(f"Modello caricato su: {self.model.device}")

        # Sottoscrizione al topic della camera
        self.image_sub = rospy.Subscriber(
            "/xtion/rgb/image_raw", Image, self.callback, queue_size=1
        )

    def callback(self, msg):
        try:
            # Conversione da ROS a OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            # Definizione classi da rilevare (es. solo cacciavite = id 3)
            # Execute prediction for specified categories on an image 0 = drill, 1 = hammer, 2 = pliers, 
            # 3 = screwdriver, 4 = wrench, 5 = saw
            CLASS = [1, 2, 3, 4]
            results = self.model.predict(frame, conf=0.25, iou=0.25, classes=CLASS)

            # Disegna predizioni
            annotated_frame = results[0].plot()

            # Mostra il risultato
            cv2.imshow("YOLO Detection", annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                rospy.signal_shutdown("Chiusura richiesta da utente")
                cv2.destroyAllWindows()

            # Se YOLO ha trovato qualcosa, pubblica il nome
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    label = self.model.names[cls_id]  # nome della classe
                    rospy.loginfo(f"Rilevato: {label}")
                    self.detection_pub.publish(label)

        except Exception as e:
            rospy.logerr(f"Errore nella callback: {e}")

    def spin(self):
        rospy.loginfo("Nodo YOLO per TIAGO avviato.")
        rospy.spin()


if __name__ == "__main__":
    detector = TiagoYoloDetector()
    detector.spin()

