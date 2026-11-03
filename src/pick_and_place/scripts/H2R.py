#!/usr/bin/env python3
import rospy, json, sys, copy, numpy as np, moveit_commander, tf,tf2_ros
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from std_msgs.msg import String, Float32, Bool

class H2RExecutor:
    def __init__(self):
        rospy.init_node("h2r_executor_node", anonymous=True)

        # MoveIt init
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.arm = moveit_commander.MoveGroupCommander("arm_torso")
        self.gripper = moveit_commander.MoveGroupCommander("gripper")

        # Parametri
        self.PREGRASP_HEIGHT = 0.20
        self.GRIPPER_OFFSET = 0.22
        self.GRIPPER_OPEN = [0.04, 0.04]
        self.TABLE_Z = 0.5

        # Carico oggetti
        json_path = "/home/pal/Desktop/DATA/picture_scene/object_poses.json"
        with open(json_path, "r") as f:
            self.objects_data = json.load(f)

        # Variabili runtime
        self.hand_pose_msg = None
        self.width = 0.04  # default, aggiornato dal topic /previous_width
        self.executing = False  # Flag per bloccare aggiornamenti hand_box durante esecuzione

        # Publisher e Subscriber
        self.h2r_done_pub = rospy.Publisher("/H2R_done", Bool, queue_size=1, latch=True)
        rospy.Subscriber("/hand_pose", PoseStamped, self.hand_pose_callback)
        rospy.Subscriber("/object2detect", String, self.object_callback)
        rospy.Subscriber("/previous_width", Float32, self.width_callback)
        

        # Aggiungo collision boxes
        self.add_collision_boxes()

    # ------------------ UTILITY ------------------
    def normalize(self, v):
        n = np.linalg.norm(v)
        return v if n < 1e-9 else v / n

    def reach_pose(self, group, pose):
        group.set_pose_target(pose)
        group.go(wait=True)
        group.stop()
        group.clear_pose_targets()

    def reach_waypoints(self, group, waypoints):
        plan, _ = group.compute_cartesian_path(waypoints, 0.01, 0.0)
        if plan:
            group.execute(plan, wait=True)
        group.stop()
        group.clear_pose_targets()

    def open_grippers(self):
        self.gripper.go(self.GRIPPER_OPEN, wait=True)
        self.gripper.stop()

    def close_grippers(self, width):
        self.gripper.go([width/2, width/2], wait=True)
        self.gripper.stop()

    def compute_pick_quaternion_y_up_z_forward_x_left(self):
        x_des = np.array([-1.0,0.0,0.0])
        y_des = np.array([0.0,1.0,0.0])
        z_des = np.array([0.0,0.0,1.0])
        R = np.eye(4)
        R[0:3,0] = x_des
        R[0:3,1] = y_des
        R[0:3,2] = z_des
        quat = tf.transformations.quaternion_from_matrix(R)
        return quat

    def compute_topdown_quaternion_x_down_align_y(self, pose_ref):
        # Prende un pose_ref e restituisce quaternion top-down x-down, y allineato a pose_ref
        q_obj = [pose_ref.orientation.x, pose_ref.orientation.y, pose_ref.orientation.z, pose_ref.orientation.w]
        R_obj = tf.transformations.quaternion_matrix(q_obj)[:3,:3]
        z_obj = R_obj[:,2]
        z_proj = np.array([z_obj[0], z_obj[1], 0.0])
        if np.linalg.norm(z_proj) < 1e-4:
            z_proj = np.array([0.0,1.0,0.0])
        z_proj = z_proj / np.linalg.norm(z_proj)
        x_des = np.array([0.0,0.0,-1.0])
        y_des = z_proj
        z_des = np.cross(x_des, y_des)
        z_des /= np.linalg.norm(z_des)
        R = np.eye(4)
        R[0:3,0] = x_des
        R[0:3,1] = y_des
        R[0:3,2] = z_des
        quat = tf.transformations.quaternion_from_matrix(R)
        return quat


    # ------------------ COLLISION BOX ------------------
    def add_collision_boxes(self):
        existing_objects = self.scene.get_known_object_names()
        if "table_box" in existing_objects:
            rospy.loginfo("table_box già presente, non riaggiungo.")
            return
        # Table box
        z_values = [obj["position"]["z"] for obj in self.objects_data.values() if "position" in obj]
        min_z = min(z_values)
        table_pose = PoseStamped()
        table_pose.header.frame_id = "base_footprint"
        table_pose.pose.position.x = min_z + 0.3
        table_pose.pose.position.y = 0
        table_pose.pose.position.z = min_z / 2
        table_pose.pose.orientation.w = 1.0
        table_size = (2*min_z, 2*min_z, min_z-0.01)
        self.scene.add_box("table_box", table_pose, size=table_size)

    # ------------------ CALLBACK ------------------
    def hand_pose_callback(self, msg):
        self.hand_pose_msg = copy.deepcopy(msg)
        hand_box_size = (0.20, 0.10, 0.05)
        hand_box_pose = PoseStamped()
        hand_box_pose.header.frame_id = "base_footprint"
        hand_box_pose.pose = msg.pose
        self.scene.add_box("hand_box", hand_box_pose, size=hand_box_size)


    def width_callback(self, msg):
        self.width = msg.data

    def object_callback(self, msg):
        side = msg.data.strip()
        
        # Ignora messaggi per R2H (nomi oggetti)
        if side not in ["Right", "Left"]:
            return  # Silenziosamente ignora, è per R2H
        
        if self.hand_pose_msg is None:
            rospy.logwarn("Attendo hand_pose prima di prelevare l'oggetto...")
            return
        
        self.pick_from_hand_and_place(side)

    
    # ------------------ PICK & PLACE ------------------
    def pick_from_hand_and_place(self, side):
        # Blocca aggiornamenti hand_box durante l'esecuzione
        self.executing = True
        
        self.open_grippers()
        rospy.loginfo("→ Movimento pre-pick sopra la mano")
        pick_pose = copy.deepcopy(self.hand_pose_msg.pose)
        pick_pose.position.x -= 0.22
        pick_pose.position.y -= 0.10
        pick_pose.position.z += 0.10  
        pick_pose.orientation = Quaternion(*self.compute_pick_quaternion_y_up_z_forward_x_left())
        self.reach_pose(self.arm, pick_pose)

        """rospy.loginfo("→ Spostamento per pick")
        grasp_pose = copy.deepcopy(pick_pose)
        grasp_pose.position.y += 0.20
        grasp_pose.orientation = pick_pose.orientation
        self.reach_waypoints(self.arm, [grasp_pose])"""

        rospy.loginfo("→ Chiusura gripper")
        self.close_grippers(self.width)

        # Sollevamento di 10cm
        rospy.loginfo("→ Sollevamento")
        post_grasp = copy.deepcopy(pick_pose)
        post_grasp.position.z += 0.10
        self.reach_waypoints(self.arm, [post_grasp])

        # Spostamento destra/sinistra
        rospy.loginfo(f"→ Spostamento verso {side}")
        lateral_pose = copy.deepcopy(post_grasp)
        lateral_pose.position.y -= 0.20 if side=="Right" else +0.20
        lateral_pose.orientation = post_grasp.orientation
        self.reach_waypoints(self.arm, [lateral_pose])


        pose_pub = rospy.Publisher("/prova_pose", PoseStamped, queue_size=1, latch=True)

        # Posa target DA SOSTITUIRE CON LA POSA DELLA SCATOLA
        pose_goal = Pose()
        pose_goal.position.x = 0.65
        pose_goal.position.y = 0.15
        pose_goal.position.z = 0.52
        pose_goal.orientation.x = -0.5052329411250166
        pose_goal.orientation.y = -0.5523567602540175
        pose_goal.orientation.z = -0.6611961551509007
        pose_goal.orientation.w = -0.04961178305117879

        # Allineamento top-down
        quat_place = self.compute_topdown_quaternion_x_down_align_y(pose_goal)
        q_place = Quaternion(*quat_place)

        # Pre-place (10 cm sopra il punto)
        pre_place = copy.deepcopy(pose_goal)
        pre_place.position.z = max(
            pose_goal.position.z + self.PREGRASP_HEIGHT + self.GRIPPER_OFFSET,
            self.TABLE_Z + self.PREGRASP_HEIGHT + 0.1
        )
        pre_place.orientation = q_place

        # Place finale (sul piano)
        final_pose = copy.deepcopy(pose_goal)
        final_pose.position.z = max(
            pose_goal.position.z + self.GRIPPER_OFFSET,
            self.TABLE_Z + self.GRIPPER_OFFSET + 0.1
        )
        final_pose.orientation = q_place


        # === Movimento verso PRE-PLACE ===
        rospy.loginfo("→ Movimento pre-place (sopra il target)")
        success = self.reach_pose(self.arm, pre_place)


        # === Discesa cartesiana verso il punto di rilascio ===
        rospy.loginfo("→ Discesa sul punto di place")
        self.reach_waypoints(self.arm, [final_pose])

        # === RILASCIO ===
        rospy.loginfo("→ Apertura gripper per rilascio")
        self.open_grippers()
        rospy.loginfo("✅ Oggetto rilasciato")

        # === RITORNO A HOME ===
        rospy.loginfo("→ Torna in home position")
        home_joints = [0.35, 0.07, 0.1, -3.1, 1.36, 2.05, 0.01, -0.05]
        self.arm.go(home_joints, wait=True)
        self.arm.stop()
        rospy.loginfo("✅ Home raggiunta")

        # Riabilita aggiornamenti hand_box
        self.executing = False

        objects_to_remove = "hand_box" # MAI rimuovere table_box!
        self.scene.remove_world_object(objects_to_remove)

        # Pubblica il segnale di completamento
        rospy.sleep(0.5)
        self.h2r_done_pub.publish(True)
        rospy.loginfo("✅ Segnale H2R_done pubblicato")
        rospy.loginfo("🔻 Spegnimento MoveIt...")
        moveit_commander.roscpp_shutdown()
        rospy.signal_shutdown("H2R complete")

# ------------------ MAIN ------------------
if __name__ == "__main__":
    executor = H2RExecutor()
    rospy.loginfo("Nodo H2R pronto. Pubblica su /hand_pose e /object2detect per iniziare.")
    rospy.spin()
