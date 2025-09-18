import numpy as np
import cv2
cap = cv2.VideoCapture(0)
colors = [(255, 0, 0), (255, 0, 255), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
color = colors[0]
previous_center = None
width = int(cap.get(3))
height = int(cap.get(4))
canvas = np.zeros((height, width, 3), np.uint8)
lower_bound = np.array([50, 100, 100])
upper_bound = np.array([90, 255, 255])
kernel = np.ones((5, 5), np.uint8)
while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (20,1), (120,65), (122,122,122), -1)
    cv2.rectangle(frame, (140,1), (220,65), colors[0], -1)
    cv2.rectangle(frame, (240,1), (320,65), colors[1], -1)
    cv2.rectangle(frame, (340,1), (420,65), colors[2], -1)
    cv2.rectangle(frame, (440,1), (520,65), colors[3], -1)
    cv2.rectangle(frame, (540,1), (620,65), colors[4], -1)
    cv2.putText(frame, "CLEAR ALL", (30, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (155, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, "VIOLET", (255, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (355, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (465, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (555, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask = cv2.dilate(mask, kernel, iterations=1)
    contours, h = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    center = None
    if len(contours) > 0:
        cmax = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cmax)
        min_area = 1000
        if area > min_area:
            M = cv2.moments(cmax)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            center = (cX, cY)
            cv2.circle(frame, center, 10, (0,0,255), 2)
            if cY < 65:
                if 20 < cX < 120:
                    canvas = np.zeros((height, width, 3), np.uint8)
                elif 140 < cX < 220:
                    color = colors[0]
                elif 240 < cX < 320:
                    color = colors[1]
                elif 340 < cX < 420:
                    color = colors[2]
                elif 440 < cX < 520:
                    color = colors[3]
                elif 540 < cX < 620:
                    color = colors[4]
            else:
                if previous_center is not None:
                    cv2.line(canvas, previous_center, center, color, 2)
                previous_center = center
        else:
            previous_center = None
    else:
        previous_center = None
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, canvas_binary = cv2.threshold(canvas_gray, 20, 255, cv2.THRESH_BINARY_INV)
    canvas_binary = cv2.cvtColor(canvas_binary, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, canvas_binary)
    frame = cv2.bitwise_or(frame, canvas)
    cv2.imshow("Air Canvas", frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
