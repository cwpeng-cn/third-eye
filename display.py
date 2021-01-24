import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
while cap.isOpened():
    ret, frame = cap.read()
    print(ret)
    index = 0
    if ret:
        index += 1
        if index % 10 == 0:
            cv2.imwrite("temp/image_{}.jpg".format(index), frame)
            print("已经捕获了:{}张".format(index))
