import cv2
import mediapipe as mp

class FaceDetector():
  def __init__(self, classifier, display_enabled=False):
    self.classifier = classifier
    self.display_image_op = display_enabled

  def _detect(self, frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_frame = cv2.GaussianBlur(gray_frame, (45, 45), cv2.BORDER_DEFAULT)
    detected_targets = self.classifier.detectMultiScale(blur_frame, scaleFactor=1.1,
                                  minNeighbors=3, minSize=(30,30), maxSize=(200,200))
    return detected_targets, blur_frame

  def find_face_box(self, frame):
    detected_faces, mod_frame = self._detect(frame)
    main_face = None
    max_area = 0
    for (x, y, w, h) in detected_faces:
      new_max_area = w * h * 0.5
      if(new_max_area > max_area):
        max_area = new_max_area
        main_face = (x, y, w, h)
      if(self.display_image_op):
        cv2.rectangle(mod_frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
    return main_face, mod_frame

class HandDetector():
  def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, track_confidence=0.5, display_enabled=False):
    self.mp_hands = mp.solutions.hands
    self.hands = self.mp_hands.Hands(mode, max_hands, detection_confidence, track_confidence)
    self.mp_draw = mp.solutions.drawing_utils
    self.display_image_op = display_enabled

  def _find_hands(self, frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    self.results = self.hands.process(frame_rgb)
    if self.results.multi_hand_landmarks:
        for hand_lms in self.results.multi_hand_landmarks:
            if self.display_image_op:
                self.mp_draw.draw_landmarks(frame_rgb, hand_lms,
                                            self.mp_hands.HAND_CONNECTIONS)
    return frame_rgb

  def find_hand_graph(self, frame, hand_idx=0):
    mod_frame = self._find_hands(frame)
    landmark_list = []
    if self.results.multi_hand_landmarks:
        selected_hand = self.results.multi_hand_landmarks[hand_idx]
        for id, lm in enumerate(selected_hand.landmark):
            h, w, c = mod_frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            landmark_list.append([id, cx, cy])
            if self.display_image_op:
                cv2.circle(mod_frame, (cx, cy), 4, (0, 255, 0), cv2.FILLED)
    return landmark_list, mod_frame
