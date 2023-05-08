import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True, tc=(255,0,255)):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxes = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox  = int(bboxC.xmin * iw), int(bboxC.ymin * ih),\
                        int(bboxC.width * iw), int(bboxC.height * ih)
                bboxes.append([id, bbox, detection.score])
                if draw:
                    img = self.fancyDraw(img,bbox)
                    cv2.putText(img, f'{int(detection.score[0]*100)}%',(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_PLAIN,
                        5, tc,2)
                
        return img, bboxes
    
    def fancyDraw(self, img, bbox, l=30, t=7, rt=1, rc=(255,0,255)):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, rc,rt)
        # Top Left
        cv2.line(img, (x, y), (x+l,y),rc,t)
        cv2.line(img, (x, y), (x,y+l),rc,t)

        # Top Right
        cv2.line(img, (x1, y), (x1-l,y),rc,t)
        cv2.line(img, (x1, y), (x1,y+l),rc,t)

        # Bottom Left
        cv2.line(img, (x, y1), (x+l,y1),rc,t)
        cv2.line(img, (x, y1), (x,y1-l),rc,t)

        # Bottom Right
        cv2.line(img, (x1, y1), (x1-1,y1),rc,t)
        cv2.line(img, (x1, y1), (x1,y1-l),rc,t)
        
        return img

def main():
    cap = cv2.VideoCapture("videos/Girl_Phone.mov")
    pTime = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img, bboxes = detector.findFaces(img)

        scale_percent = 25  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        imgToShow = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(imgToShow, f'FPS {int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,
                    3, (0,255,0),2)
        cv2.imshow("Image", imgToShow)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()