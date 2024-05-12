import cv2

def main(url):
    camera = cv2.VideoCapture(url)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # 1080, 720, 360
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # 1920, 1280, 640
    camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    camera.set(cv2.CAP_PROP_FOCUS, 0)
    succes, img = camera.read()
    #img = cv2.imread(url)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(10, 10))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    eimg = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    HSVImg = cv2.cvtColor(eimg, cv2.COLOR_BGR2HSV)
    robots = []
    for i in range(1):
        grayImg = cv2.inRange(HSVImg, HSVMin[i], HSVMax[i])
        ret, thresh = cv2.threshold(grayImg, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        num = 0
        for c in contours:
            if cv2.contourArea(c) >= squareCells[i]:
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                robots.append([cX, cY])
                cv2.circle(img, (cX, cY), 53, (0, 0, 255), 2)
                cv2.putText(img, f'id: {num}', (cX + 50, cY - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                num += 1
    cv2.imshow(f'Img-{url}', img)
    cv2.imwrite('output.png', img)
    while cv2.waitKey(1) != 27: pass
    cv2.destroyAllWindows()


if __name__ == '__main__':
    squareCells = [270, 3700]
    #HSVMin = [(118, 255, 45), (0, 0, 0)]
    #HSVMax = [(139, 255, 255), (0, 0, 0)]
    #реальные значения
    #(35, 14, 158) (118, 61, 250)]
    HSVMin = [(52, 59, 43), (0, 9, 0)]
    HSVMax = [(160, 167, 102), (203, 99, 94)]
    urls = [2]
    for url in urls: main(url)
    cv2.destroyAllWindows()