import cv2

for i in range(101):
	cap = cv2.VideoCapture(i)
	if not cap.isOpened():
	    print("Cannot open camera")
	else:
	    ret, frame = cap.read()
	    cv2.imwrite(f'cam{i}-t2.png', frame)
	    cv2.imshow('cam', frame)
	    while cv2.waitKey(1) != 27: pass
	    cap.release()
	    cv2.destroyAllWindows()
