#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import open3d as o3d
from ultralytics import YOLO
import os
from scipy.spatial.transform import Rotation as R
import tf
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo
from sklearn.cluster import DBSCAN
import json
from std_srvs.srv import Trigger, TriggerResponse
from std_msgs.msg import Bool
import threading


class ObjectLocalizationNode:
    def __init__(self):
        rospy.init_node("object_localization", anonymous=True)
        rospy.loginfo("=== Object Localization Node Started ===")

        # === PARAMETRI ===
        self.folder = rospy.get_param("~data_folder", "/home/pal/Desktop/DATA/picture_scene")
        self.model_path = rospy.get_param("~model_path", "/home/pal/IRIM_manipulation_ws/src/object_recognition/scripts/prova.pt")
        self.file_trans = os.path.join(self.folder, "transform_info.txt")

        # === CARICA CAMERA INFO ===
        # Invece di aspettare camera_info dal topic, usa una matrice K hardcoded
        # o caricala dal primo nodo se disponibile
        self.width, self.height, self.K = self.get_camera_info_from_file()

        # === YOLO MODEL ===
        self.model = YOLO(self.model_path)
        self.model.to("cuda")
        self.CLASS = [2, 3, 4]

        # === MEMORIA OGGETTI ===
        self.objects_pose_dict = {}
        self.objects_camera_dict = {}
        self.publishers = {}

        # === PUBLISHER BOOLEANO FINE LOCALIZZAZIONE ===
        self.pub_done = rospy.Publisher('/object_localization_done', Bool, queue_size=1)

        rospy.loginfo("Nodo pronto, in attesa di richiesta per la localizzazione oggetti.")

    # ---------------------- CAMERA INFO ----------------------
    def get_camera_info_from_file(self):
        """Carica la matrice K da un file o usa valori di default"""
        camera_info_file = os.path.join(self.folder, "camera_info.txt")

        # Prova a caricare da file se esiste
        if os.path.exists(camera_info_file):
            try:
                with open(camera_info_file, "r") as f:
                    lines = f.readlines()
                    if len(lines) >= 3:
                        # Prima riga: width, height
                        width, height = map(int, lines[0].strip().split(","))
                        # Righe successive: matrice K (3 righe da 3 valori ciascuna)
                        K = []
                        for i in range(3):
                            if i+1 < len(lines):
                                row = list(map(float, lines[i+1].strip().split(",")))
                                K.extend(row)
                        K = np.array(K).reshape(3, 3)
                        rospy.loginfo(f"Loaded camera info from file: {camera_info_file}")
                        return width, height, K
            except Exception as e:
                rospy.logwarn(f"Failed to load camera info from file: {e}")

        # Valori di default per Xtion RGB camera
        # Questi sono valori tipici per una camera Xtion RGB
        width = 640
        height = 480
        fx = 570.3422241210938   # focal length x
        fy = 570.3422241210938   # focal length y
        cx = 314.5               # principal point x
        cy = 239.5               # principal point y

        K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ])

        rospy.loginfo(f"Using default camera info - Width: {width}, Height: {height}")
        rospy.loginfo(f"K matrix:\n{K}")
        return width, height, K

    # ---------------------- FILE HANDLING ----------------------
    def read_transformation(self, file_path):
        trans_list = []
        with open(file_path, "r") as f:
            for line in f:
                values = [float(v) for v in line.strip().split(",")]
                trans_list.append(values)
        return trans_list

    def load_data(self, folder):
        png_files, pcd_files = [], []
        for root, _, filenames in os.walk(folder):
            for filename in filenames:
                if filename.lower().endswith(".png"):
                    png_files.append(os.path.join(root, filename))
                elif filename.lower().endswith(".pcd"):
                    pcd_files.append(os.path.join(root, filename))
        return png_files, pcd_files

    # ---------------------- SEGMENTAZIONE ----------------------
    def mask_segmentation(self, cropped):
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        m = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if num_labels <= 1:
            return m
        else:
            areas = stats[1:, cv2.CC_STAT_AREA]
            best = 1 + np.argmax(areas)
            mask = (labels == best).astype(np.uint8) * 255
        return mask

    # ---------------------- PCA e ROTAZIONE ----------------------
    def compute_pca_axis_3d(self, points):

        if len(points) < 10:
            return None, None, None, None, None

        pts = np.asarray(points)
        centroid = np.mean(pts, axis=0)

        # PCA
        cov = np.cov((pts - centroid).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]

        # Asse principale (lunghezza)
        main_axis = eigvecs[:, 0]
        # Asse perpendicolare (larghezza)
        perp_axis = eigvecs[:, 1]

        # Proiezione dei punti lungo asse perpendicolare
        proj = (pts - centroid) @ perp_axis
        pmin = pts[np.argmin(proj)]
        pmax = pts[np.argmax(proj)]

        # Larghezza 3D (in metri)
        width_3d = np.linalg.norm(pmax - pmin)

        return centroid, eigvecs, width_3d

    def build_rotation_from_axis(self, main_axis, force_z_positive=True):
        z_obj = main_axis / np.linalg.norm(main_axis)
        if force_z_positive and z_obj[2] < 0:
            z_obj = -z_obj
        up_img = np.array([0, -1, 0], dtype=float)
        x_obj = np.cross(up_img, z_obj)
        if np.linalg.norm(x_obj) < 1e-6:
            up_img = np.array([1, 0, 0], dtype=float)
            x_obj = np.cross(up_img, z_obj)
        x_obj /= np.linalg.norm(x_obj)
        y_obj = np.cross(z_obj, x_obj)
        y_obj /= np.linalg.norm(y_obj)
        return np.column_stack((x_obj, y_obj, z_obj))

    def rotation_to_quaternion(self, R_mat):
        r = R.from_matrix(R_mat)
        return r.as_quat()  # [x, y, z, w]

    # ---------------------- LOCALIZZAZIONE ----------------------
    def run_localization(self):
        detections = []
        detections_camera = []

        self.png_files, self.pcd_files = self.load_data(self.folder)
        self.transformations = self.read_transformation(self.file_trans)

        for f in range(len(self.png_files)):
            rospy.loginfo(f"Processing frame {f}/{len(self.png_files)}")

            png_file = os.path.join(self.folder, f"rgb_image_color_{f}.png")
            pcd_file = os.path.join(self.folder, f"depth_registered_points{f}.pcd")

            frame = cv2.imread(png_file)
            pcd = o3d.io.read_point_cloud(pcd_file)
            points = np.asarray(pcd.points)
            results = self.model.predict(frame, conf=0.45, iou=0.45, classes=self.CLASS, agnostic_nms= True) 
            # annotated_frame=results[0].plot()
            # cv2.imshow("detection", annotated_frame)
            # cv2.waitKey(0)  # Commentato per esecuzione automatica

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls = int(box.cls[0].item())

                x1 = max(0, int(x1) - 10)
                y1 = max(0, int(y1) - 10)
                x2 = min(frame.shape[1], int(x2) + 10)
                y2 = min(frame.shape[0], int(y2) + 10)
                cropped = frame[y1:y2, x1:x2]
                mask_2D = self.mask_segmentation(cropped)
                if mask_2D is None:
                    continue

                mask_full = np.zeros(frame.shape[:2], dtype=np.uint8)
                mask_full[y1:y2, x1:x2] = mask_2D

                if cls == 3:
                    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
                    masked_hsv = hsv[mask_2D > 0]
                    if len(masked_hsv) > 0:
                        h_mean = np.mean(masked_hsv[:, 0])
                        if h_mean < 60 or h_mean >= 140:
                            cls = "screwdriver_red"
                        else:
                            cls = "screwdriver_black"

                obj_points = []
                for pt in points:
                    X, Y, Z = pt
                    if Z <= 0:
                        continue
                    u = int((X * self.K[0, 0]) / Z + self.K[0, 2])
                    v = int((Y * self.K[1, 1]) / Z + self.K[1, 2])
                    if 0 <= u < self.width and 0 <= v < self.height:
                        if mask_full[v, u] > 0:
                            obj_points.append(pt)

                if len(obj_points) == 0:
                    continue

                obj_points = np.array(obj_points)
                centroid, eigvecs, width_3d = self.compute_pca_axis_3d(obj_points)
                if centroid is None:
                    continue
                main_axis = eigvecs[:, 0]
                R_mat = self.build_rotation_from_axis(main_axis)
                quat = self.rotation_to_quaternion(R_mat)
                mask_area = cv2.countNonZero(mask_2D)

                transform = self.transformations[f]
                trans = np.array(transform[0:3])
                rot = np.array(transform[3:])
                R_base = tf.transformations.quaternion_matrix(rot)[:3, :3]
                position_base = R_base.dot(centroid) + trans
                quat_base = tf.transformations.quaternion_multiply(rot, quat)

                detections.append({
                    "cls": cls,
                    "pos": position_base,
                    "quat": quat_base,
                    "mask_area": mask_area,
                    "obj_width": width_3d
                })

                detections_camera.append({
                    "cls": cls,
                    "centroid": centroid,
                    "quat": quat,
                    "mask_area": mask_area,
                    "obj_width": width_3d
                })

        # === CLUSTERING ===
        final_objects = []
        final_objects_camera = []

        for cls in set(d["cls"] for d in detections):
            cls_dets = [d for d in detections if d["cls"] == cls]
            cls_dets_cam = [d for d in detections_camera if d["cls"] == cls]
            centroids = np.array([d["pos"] for d in cls_dets])
            if len(centroids) == 0:
                continue
            clustering = DBSCAN(eps=0.5, min_samples=1).fit(centroids)
            labels = clustering.labels_
            for label in np.unique(labels):
                cluster_dets = [cls_dets[i] for i in range(len(cls_dets)) if labels[i] == label]
                cluster_dets_cam = [cls_dets_cam[i] for i in range(len(cls_dets_cam)) if labels[i] == label]

                # --- Controllo larghezza per pliers e wrench ---
                if cls in [2, 4]:  # 2=pliers, 4=wrench
                    widths = [d["obj_width"] for d in cluster_dets if 0.02 < d["obj_width"] < 0.15]
                    if len(widths) == 0:
                        rospy.logwarn(f"Nessuna width valida per {self.get_class_name(cls)}, uso la più piccola trovata.")
                        widths = [min(d["obj_width"] for d in cluster_dets)]
                    realistic_width = np.median(widths)
                    # Aggiorna tutti i detections del cluster con questa width filtrata
                    for d in cluster_dets:
                        d["obj_width"] = realistic_width
                    for d in cluster_dets_cam:
                        d["obj_width"] = realistic_width

                # --- Seleziona detection migliore per area ---
                best_det = max(cluster_dets, key=lambda d: d["mask_area"])
                best_det_cam = max(cluster_dets_cam, key=lambda d: d["mask_area"])
                final_objects.append(best_det)
                final_objects_camera.append(best_det_cam)
                
        # === PUBBLICAZIONE E SALVATAGGIO ===
        for obj, obj_cam in zip(final_objects, final_objects_camera):
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = "base_footprint"
            pose.pose.position.x = obj["pos"][0]
            pose.pose.position.y = obj["pos"][1]
            pose.pose.position.z = obj["pos"][2]
            pose.pose.orientation.x = obj["quat"][0]
            pose.pose.orientation.y = obj["quat"][1]
            pose.pose.orientation.z = obj["quat"][2]
            pose.pose.orientation.w = obj["quat"][3]

            class_name = f"{self.get_class_name(obj['cls'])}"
            width = obj["obj_width"]
            topic = f"/{class_name}"

            if topic not in self.publishers:
                self.publishers[topic] = rospy.Publisher(topic, PoseStamped, queue_size=10)
            self.publishers[topic].publish(pose)
            self.objects_pose_dict[class_name] = {
                "pose": pose,
                "width": width
            }

            self.objects_camera_dict[class_name] = {
                "centroid": {
                    "x": obj_cam["centroid"][0],
                    "y": obj_cam["centroid"][1],
                    "z": obj_cam["centroid"][2]
                },
                "orientation": {
                    "x": obj_cam["quat"][0],
                    "y": obj_cam["quat"][1],
                    "z": obj_cam["quat"][2],
                    "w": obj_cam["quat"][3]
                },
                "object_width": obj_cam["obj_width"],
                "color": class_name.split("_")[-1]
            }

        self.save_pose_dict("/home/pal/Desktop/DATA/picture_scene/object_poses.json")
        self.save_camera_dict("/home/pal/Desktop/DATA/picture_scene/object_poses_camera.json")
        rospy.loginfo("Object localization completata!")

        # === PUBBLICO BOOLEANO COMPLETAMENTO ===
        self.pub_done.publish(True)

    # ---------------------- CLASSI e SALVATAGGIO ----------------------
    def get_class_name(self, cls_id):
        mapping = {
            2: "pliers",
            3: "screwdriver",
            4: "wrench"
        }
        return mapping.get(cls_id, f"{cls_id}")

    def save_pose_dict(self, filename):
        data = {}
        for name, info in self.objects_pose_dict.items():
            pose = info["pose"]
            width = info["width"]
            data[name] = {
                "position": {
                    "x": pose.pose.position.x,
                    "y": pose.pose.position.y,
                    "z": pose.pose.position.z
                },
                "orientation": {
                    "x": pose.pose.orientation.x,
                    "y": pose.pose.orientation.y,
                    "z": pose.pose.orientation.z,
                    "w": pose.pose.orientation.w
                },
                "width": width,
                "color": name.split("_")[-1]
            }
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        rospy.loginfo(f"Pose dictionary saved to {filename}")

    def save_camera_dict(self, filename):
        with open(filename, "w") as f:
            json.dump(self.objects_camera_dict, f, indent=4)
        rospy.loginfo(f"Camera centroid dictionary saved to {filename}")


if __name__ == "__main__":
    try:
        node = ObjectLocalizationNode()

        def handle_localization(req):
            rospy.loginfo("Avvio localizzazione oggetti...")

            def run_and_publish():
                node.run_localization()
                rospy.loginfo("Localizzazione completata e booleano pubblicato.")
                rospy.signal_shutdown("Localizzazione completata")

            t = threading.Thread(target=run_and_publish)
            t.start()
            return TriggerResponse(success=True, message="Localizzazione avviata.")

        srv = rospy.Service("/run_object_localization", Trigger, handle_localization)
        rospy.loginfo("Service /run_object_localization pronto.")
        rospy.spin()

    except rospy.ROSInterruptException:
        pass