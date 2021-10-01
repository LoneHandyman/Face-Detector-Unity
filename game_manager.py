import numpy
import socket

def sigmoid(x):
  return (1 / (1 + numpy.exp(-x))) * 2 - 1

class FaceCamManagerUnity:
  def __init__(self, remote_ip, remote_port):
    self.last_input = 0.0
    self.remote_ip = remote_ip
    self.remote_port = remote_port
    self.udp_connection = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

  def keepConection(self):
    #Si se acaba el juego, unity mandará una señal para detener el detector
    return True

  def send_face_cam_input(self, face_box):
    if face_box is not None:
      x_vel = float(face_box[0]) + float(face_box[2]) * 0.5
      normalized_x_vel = round(sigmoid(x_vel - self.last_input), 4)
      self.last_input = x_vel
      print(normalized_x_vel)
      #self.udp_connection.sendto(str(normalized_x_vel), (self.remote_ip, self.remote_port))