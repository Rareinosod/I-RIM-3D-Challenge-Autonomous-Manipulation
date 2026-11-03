#!/usr/bin/env python3
import rospy
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped
import cv2
from cv_bridge import CvBridge
import mediapipe as mp
import random

bridge = CvBridge()
gesture_enabled = True  # default True, state machine lo gestisce
selection_locked = False  # reset when gesture re-enabled
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

VALID_GESTURES = {"screwdriver_green", "screwdriver_red", "screwdriver_black",
                  "hammer", "wrench", "pliers", "Left", "Right"}

def enable_callback(msg):
    global gesture_enabled, selection_locked
    gesture_enabled = msg.data
    if gesture_enabled:
        # Reset selection_locked quando gesture viene riabilitato
        selection_locked = False
    rospy.loginfo(f"Gesture recognition {'ENABLED' if gesture_enabled else 'DISABLED'}")

def fingers_up(landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    pip_ids = [3, 6, 10, 14, 18]
    fingers = [False] * 5
    if landmarks[tips_ids[0]][0] < landmarks[2][0]:
        fingers[0] = True
    for i in range(1, 5):
        if landmarks[tips_ids[i]][1] < landmarks[pip_ids[i]][1]:
            fingers[i] = True
    return fingers

def classify_gesture(fingers):
    thumb, index, middle, ring, pinky = fingers
    if thumb and not any([index, middle, ring, pinky]): return "screwdriver_green"
    if thumb and index and not any([middle, ring, pinky]): return "screwdriver_red"
    if thumb and index and middle and not any([ring, pinky]): return "screwdriver_black"
    if all(fingers): return "hammer"
    if index and pinky and not any([thumb, middle, ring]): return "wrench"
    if thumb and pinky and not any([index, middle, ring]): return "pliers"
    if not any(fingers): return "Right"
    return None

def main():
    global gesture_enabled, selection_locked
    rospy.init_node("gesture_node")
    rospy.Subscriber("/gesture_enable", Bool, enable_callback)
    object_pub = rospy.Publisher("/object2detect", String, queue_size=10, latch=True)  # LATCH per mantenere ultimo messaggio
    hand_pose_pub = rospy.Publisher("/hand_pose", PoseStamped, queue_size=10, latch=True)

    cap = cv2.VideoCapture(0)
    rate = rospy.Rate(10)
    last_gesture = None
    countdown = 0
    frame_threshold = 20  # Stesso valore del nodo reale
    confirmed_gesture = None
    
    # Variabile per mantenere l'ultimo hand_pose
    last_hand_pose = PoseStamped()
    last_hand_pose.header.frame_id = "base_footprint"
    last_hand_pose.pose.position.x = 0.8
    last_hand_pose.pose.position.y = 0.3
    last_hand_pose.pose.position.z = 0.8
    last_hand_pose.pose.orientation.w = 1.0

    while True:

        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                        min_detection_confidence=0.6, min_tracking_confidence=0.6)

        while not rospy.is_shutdown():
            ret, frame = cap.read()
            if not ret:
                rate.sleep()
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Disegna i landmark sulla mano
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Aggiorna hand_pose con movimento simulato
                    current_hand_pose=PoseStamped()
                    current_hand_pose.header.frame_id = "base_footprint"
                    current_hand_pose.header.stamp = rospy.Time.now()
                    current_hand_pose.pose.position.x = 0.8 + random.uniform(-0.05, 0.05)
                    current_hand_pose.pose.position.y = 0.3 + random.uniform(-0.05, 0.05)
                    current_hand_pose.pose.position.z = 0.8 + random.uniform(-0.05, 0.05)
                    current_hand_pose.pose.orientation.w = 1.0
                    
                    # Gesture recognition solo se abilitata
                    if gesture_enabled:
                        last_hand_pose = current_hand_pose
                        lm = [(l.x, l.y) for l in hand_landmarks.landmark]
                        fingers = fingers_up(lm)
                        gesture = classify_gesture(fingers)
                        if gesture in VALID_GESTURES and not selection_locked:
                            if gesture == last_gesture:
                                countdown += 1
                            else:
                                countdown = 1
                                last_gesture = gesture
                            if countdown >= frame_threshold:
                                confirmed_gesture = gesture
                                selection_locked = True
                                object_pub.publish(confirmed_gesture)
                                rospy.loginfo(f"✅ Gesto confermato: {confirmed_gesture}")
                        else:
                            countdown = 0
                            last_gesture = None
            
            # Pubblica sempre l'ultimo hand_pose, anche se gesture è disabilitato
            hand_pose_pub.publish(last_hand_pose)
            
            # Mostra stato sulla finestra
            status_text = f"Gesture: {'ENABLED' if gesture_enabled else 'DISABLED'}"
            color = (0, 255, 0) if gesture_enabled else (0, 0, 255)
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            if last_gesture and gesture_enabled:
                cv2.putText(frame, f"Detecting: {last_gesture} ({countdown}/{frame_threshold})", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if confirmed_gesture:
                cv2.putText(frame, f"Confirmed: {confirmed_gesture}", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Gesture Recognition - PC Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            rate.sleep()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
