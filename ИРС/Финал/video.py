import cv2
from time import time
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('your_video.avi', fourcc, 20.0, size)

t = time() + 5
while t > time():
    _, frame = cap.read()
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
