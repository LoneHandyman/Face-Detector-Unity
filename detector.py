import cv2
import game_manager

class FaceCamDetector:
  def __init__(self, unity_server_address, display_enabled=False):
    self.manager = game_manager.FaceCamManagerUnity(unity_server_address[0], unity_server_address[1])
    self.camera_device = cv2.VideoCapture(0)
    self.display_image_op = display_enabled
    if not (self.camera_device.isOpened()):
      print("Error: Could not open your camera device.")
      exit(1)
    self.classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")

  def _detect_face(self, frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_frame = cv2.GaussianBlur(gray_frame, (45, 45), cv2.BORDER_DEFAULT)
    detected_faces = self.classifier.detectMultiScale(blur_frame, scaleFactor=1.1,
                                minNeighbors=3, minSize=(30,30), maxSize=(200,200))
    main_face = None
    max_area = 0
    for (x, y, w, h) in detected_faces:
      new_max_area = w * h * 0.5
      if(new_max_area > max_area):
        max_area = new_max_area
        main_face = (x, y, w, h)
      if(self.display_image_op):
        cv2.rectangle(blur_frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
    if(self.display_image_op):
      cv2.imshow('FaceCamDetector', blur_frame)
    return main_face

  def update_frame_by_frame(self):
    self.camera_device.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    self.camera_device.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while(self.manager.keepConection()):
      retval, frame = self.camera_device.read()
      if(retval):
        self.manager.send_face_cam_input(self._detect_face(frame))
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  def destroy(self):
    self.camera_device.release()
    cv2.destroyAllWindows()


detector = FaceCamDetector(("127.0.0.1", 5001), True)
detector.update_frame_by_frame()
detector.destroy()