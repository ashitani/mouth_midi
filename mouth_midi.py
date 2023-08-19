import cv2
import numpy as np
import mediapipe as mp
import mido
import time

class MouthMidiSender:

    left_eye_indexes = [7, 163, 144, 145, 153, 154,
                        155, 173, 157, 158, 159, 160, 161, 246]
    right_eye_indexes = [381, 380, 374, 373, 390, 249,
                         466, 388, 387, 386, 385, 384, 398, 382, 381]
    eyes_farest_indexes = [7, 466]
    mouth_indexes = [13, 14]

    small_window_size = (320, 180)
    movie_window_size = (500, 180)

    def __init__(self, port_index, cc_channels,
                 sensitivity=1.0, polarity=1, mosaic_size=100,
                 record_file=None):

        self.port = self.select_port(port_index)
        self.cc_channels = cc_channels
        self.sensitivity = np.clip(sensitivity, 0.1, 5.0)
        self.polarity = polarity

        self.mosaic_size = mosaic_size

        self.record_file = record_file
        self.record = (self.record_file != None)

        self.init_mediapipe()
        self.midiout = mido.open_output(self.port)

    def select_port(self, port_index):
        ports = mido.get_output_names()
        port = ports[port_index]
        print("MIDI output ports:")
        for i, p in enumerate(ports):
            if i == port_index:
                print("*%d : %s" % (i, p))
            else:
                print(" %d : %s" % (i, p))
        print()
        return port

    def init_mediapipe(self):
        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.cap = cv2.VideoCapture(1)

    def loop(self):
        self.oldval = 0
        self.oldtime = 0
        if (self.record):
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            name = self.record_file
            self.video = cv2.VideoWriter(
                name, fourcc, 24, self.movie_window_size)
            print("output video is written in "+name)

        while self.cap.isOpened():
            # print_fps()

            # get image from camera
            success, image = self.cap.read()
            if not success:
                continue
            image = cv2.flip(image, 1)

            # get size of mouth_and send midi
            keys = self.get_keys(image)
            val = self.get_mouth_size(keys)
            self.send_midi(val)

            # visualize
            self.visualize(image, val, keys)

            # escape key to break
            if cv2.waitKey(5) & 0xFF == 27:
                break

        self.face_mesh.close()
        self.cap.release()

        if self.record:
            self.video.release()

    def print_fps(self):
        t = time.time()
        if(t != self.oldtime):
            fps = 1.0/(t-self.oldtime)
            print(fps)
        self.oldtime = t

    def send_midi(self, val):
        if(val != self.oldval and val != -1):
            cc = mido.Message('control_change', channel=1,
                              control=1, value=val, time=60)
            self.midiout.send(cc)
        oldval = self.oldval

    def draw_val(self, img, val):
        bar_size = 120*(val/127.0)
        cv2.rectangle(img,
                      pt1=(140, 140),
                      pt2=(160, int(140-bar_size)),
                      color=(0, 0, 255),
                      thickness=-1,
                      lineType=cv2.LINE_4)

        cv2.putText(img, "%03d" % val, (120, 170), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,
                    color=(0, 0, 255),
                    thickness=1,
                    lineType=cv2.LINE_4)
        return img

    def get_mouth_size(self, keys, min_dist=0.01, max_dist=0.125):

        min_dist = min_dist
        max_dist = max_dist/self.sensitivity

        val = 0
        if (13 in keys) and (14 in keys):
            up = np.array(keys[13])
            down = np.array(keys[14])
            dist = np.linalg.norm(up-down) - min_dist

            dist = np.clip(dist, 0, max_dist-min_dist)
            val = int(dist/(max_dist-min_dist) * 127)

        if(self.polarity != 1):
            val = 127-val

        return val

    def get_keys(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.face_mesh.process(image)

        keys = {}
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    keys[idx] = [lm.x, lm.y, lm.z]
        return keys

    def draw_part(self, image, keys, target_keys, col, circle_size=1):
        positions = self.get_key_positions(image, keys, target_keys)
        for pos in positions:
            cv2.circle(image, pos, circle_size, col, -1)

        return image

    def draw_mosaic(self, image, keys, target_keys, col, line_width=10):
        positions = self.get_key_positions(image, keys, target_keys)
        if(len(positions)==2):
            cv2.line(image, positions[0], positions[1],
                    col, line_width, cv2.LINE_4)
        return image

    def get_key_positions(self, image, keys, target_keys):
        h, w, c = np.shape(image)
        positions = []
        for i in target_keys:
            if i in keys:
                lm = keys[i]
                pos = (int(lm[0]*w), int(lm[1]*h))
                positions.append(pos)
        return positions

    def draw_parts(self, image, keys):
        #    image = self.draw_part(image, keys, self.left_eye_indexes, (0,0,255),8)
        #    image = self.draw_part(image, keys, self.right_eye_indexes, (0,255,0),8)
        if(self.mosaic_size != 0):
            image = self.draw_mosaic(
                image, keys, self.eyes_farest_indexes, (0, 0, 0), self.mosaic_size)
        image = self.draw_part(image, keys, self.mouth_indexes, (0, 0, 255), 4)

        return image

    def get_mouth_image(self, image, keys):
        positions = self.get_key_positions(image, keys, self.mouth_indexes)
        if len(positions)==2:
          center = (
              int((positions[0][0]+positions[1][0])/2),
              int((positions[0][1]+positions[1][1])/2)
          )
          height,width,_ = np.shape(image)
          t= np.clip(int(center[1]-280),0,height)
          b= np.clip(int(center[1]+80),0,height)
          l= np.clip(int(center[0]-100),0,width)
          r= np.clip(int(center[0]+100),0,width)
          mouth_img = image[t:b, l:r]
          rate=180.0/np.shape(mouth_img)[0]
          mouth_img = cv2.resize(mouth_img,None,fx=rate,fy=rate)
          mouth_img = cv2.resize(mouth_img,(100,180))
          mouth_img = np.hstack((mouth_img,np.zeros((180,80,3),dtype=np.uint8)))
        else:
          mouth_img=np.zeros((180,180,3),dtype=np.uint8)
        return mouth_img

    def visualize(self, image, val, keys):

        # mouth image window
        image = self.draw_parts(image, keys)
        mouth_image = self.get_mouth_image(image, keys)
        mouth_image = self.draw_val(mouth_image,val)
#        cv2.imshow('Mouth',mouth_image)

        # small image window
        small = cv2.resize(image, self.small_window_size)
        small = np.hstack((small,mouth_image))
        cv2.imshow('MouthMIDI', small)


        # record movie
        if (self.record):
            self.video.write(small)

if __name__ == "__main__":

    port_index = 0              # MIDI output port number
    cc_channels = [1]           # MIDI CC channel
    sensitivity = 2.5           # sensitivity from mouth size to CC value
    polarity = 1                # MIDI polarity
    mosaic_size = 0             # if 0, eye mosaic is disabled
    record_file = "sample.mp4"  # if None, record is disabled

    sender = MouthMidiSender(
        port_index, cc_channels, sensitivity, polarity, mosaic_size, record_file)
    sender.loop()
