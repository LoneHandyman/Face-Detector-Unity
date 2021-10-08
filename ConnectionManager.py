import copy
import cv2
import math
import numpy
import socket
import OpenCvObjDetectors as trackers

sigmoid = lambda x: 1 / (1 + numpy.exp(-x))
FLOAT_PRECISION = 4

class ConnectionManager():
    def __init__(self, unity_server_ip, unity_server_port, draw_enabled=False):
        self.FACE_CLASSIFIER_NAME = "haarcascade_frontalface_alt.xml"
        self.camera_device = cv2.VideoCapture(0)
        if not (self.camera_device.isOpened()):
            print("<Error>: Could not open your camera device.")
            exit(1)
        self.camera_device.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera_device.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        classifier_path = cv2.data.haarcascades + self.FACE_CLASSIFIER_NAME
        self.face_classifier = cv2.CascadeClassifier(classifier_path)
        print("<Source>:", self.face_classifier)
        print("<Path>:", classifier_path)

        self.draw_enabled = draw_enabled
        self.unity_server_ip = unity_server_ip
        self.unity_server_port = unity_server_port
        self.tcp_connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_connection.connect((self.unity_server_ip, self.unity_server_port))
        self.data_face_acceleration = [0.0, 0.0]

    def run(self):
        face_detector = trackers.FaceDetector(classifier=self.face_classifier, display_enabled=self.draw_enabled)
        hand_detector = trackers.HandDetector(detection_confidence=0.7, display_enabled=self.draw_enabled)
        last_x_factor_value = 0.0
        while (True):
            retval, frame = self.camera_device.read()
            if (retval):
                copy_frame = copy.copy(frame)
                box, mod_face_frame = face_detector.find_face_box(frame)
                landmarks, mod_hand_frame = hand_detector.find_hand_graph(copy_frame)

                norm_movement_fact = norm_acceleration_fact = 0.0
                if box is not None:
                    x_vel = float(box[0]) + float(box[2]) * 0.5
                    norm_movement_fact = round(sigmoid(x_vel - last_x_factor_value) * 2 - 1, FLOAT_PRECISION)
                    last_x_factor_value = x_vel
                    # print("<Movement_Factor>: %7.4f" % (norm_movement_fact))

                if len(landmarks) != 0:
                    x1, y1 = landmarks[4][1], landmarks[4][2]
                    x2, y2 = landmarks[8][1], landmarks[8][2]
                    norm_acceleration_fact = round(min(math.hypot(x2 - x1, y2 - y1), 160.0) / 160.0, FLOAT_PRECISION)
                    # print("<Acceleration Factor>: %7.4f" % (norm_acceleration_fact))
                # self.udp_connection.sendto(str(norm_movement_fact) + " " + str(norm_acceleration_fact), (self.unity_server_ip, self.unity_server_port))

                self.data_face_acceleration = [norm_movement_fact, norm_acceleration_fact]
                postData = ','.join(map(str, self.data_face_acceleration))
                self.tcp_connection.sendall(postData.encode("UTF-8"))

                if (self.draw_enabled):
                    cv2.imshow("Face", mod_face_frame)
                    cv2.imshow("Hand", mod_hand_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    demo = ConnectionManager("127.0.0.1", 50001, True)
    demo.run()
