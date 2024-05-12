import cv2
from time import time
cap1 = cv2.VideoCapture(8)
cap2 = cv2.VideoCapture(4)
cap3 = cv2.VideoCapture(0)
width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out1 = cv2.VideoWriter('video-1.avi', fourcc, 20.0, size)
out2 = cv2.VideoWriter('video-2.avi', fourcc, 20.0, size)
out3 = cv2.VideoWriter('video-3.avi', fourcc, 20.0, size)

t = time() + 5
while t > time(): r, img1 = cap1.read()
t = time() + 5
while t > time(): r, img1 = cap2.read()
t = time() + 5
while t > time(): r, img1 = cap3.read()
t = time() + 15
while t > time():
    r, img1 = cap1.read()
    r, img2 = cap2.read()
    r, img3 = cap3.read()
    out1.write(img1)
    out2.write(img2)
    out3.write(img3)

cap1.release()
cap2.release()
cap3.release()
out1.release()
out2.release()
out3.release()
cv2.destroyAllWindows()
