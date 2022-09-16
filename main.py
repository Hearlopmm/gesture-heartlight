import cv2
import mediapipe as mp
import serial

# ser = serial.Serial('COM3', 115200, timeout=0.5)
# ser.open()
# flag = ser.is_open

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils
results = 0
lcount = 0
rcount = 0
pList = []
flag = False

pTime = 0
cTime = 0

c = 0x00


# 串口打开函数
def open_ser():
    port = 'COM7'  # 串口号
    baudrate = 115200  # 波特率
    try:
        global ser
        global flag
        ser = serial.Serial(port, baudrate, timeout=0.05)
        if ser.isOpen():
            flag = True
            print("串口打开成功")
    except Exception as exc:
        print("串口打开异常", exc)


def findHands(img, draw=True):  # whether draw key points
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    global results
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            if draw:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    return img


def findPosition(img, draw=True):  # draw:whether write labels beside key points
    lmLists = []
    imgHeight = img.shape[0]
    imgWidth = img.shape[1]

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for i, lm in enumerate(handLms.landmark):
                xPos = int(lm.x * imgWidth)
                yPos = int(lm.y * imgHeight)
                lmLists.append([i, xPos, yPos])
                if draw:
                    cv2.putText(img, str(i), (xPos - 25, yPos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 7, 137), 2)

    return lmLists


def judgeHands(lmList, draw=True):  # draw:whether write the result
    count = 0
    global c
    if lmList[16][1] < lmList[12][1] < lmList[8][1] and lmList[20][2] > lmList[13][2] and \
            lmList[4][2] > lmList[5][2] and lmList[14][2] > lmList[15][2] > lmList[16][2] and \
            lmList[10][2] > lmList[11][2] > lmList[12][2] and lmList[6][2] > lmList[7][2] > lmList[8][2] and \
            lmList[12][2] < lmList[0][2]:
        count = 3
        c = 0x03
    elif lmList[20][1] < lmList[16][1] < lmList[12][1] and lmList[18][2] > lmList[19][2] > \
            lmList[20][2] and lmList[14][2] > lmList[15][2] > lmList[16][2] and lmList[4][2] > lmList[9][2] and \
            lmList[8][2] > lmList[5][2] and lmList[12][2] < lmList[0][2]:
        count = 3
        c = 0x03
    elif lmList[8][2] < lmList[7][2] < lmList[6][2] < \
            min(lmList[20][2], lmList[16][2], lmList[12][2], lmList[4][2]):
        count = 1
        c = 0x01
    elif lmList[8][2] < lmList[7][2] < lmList[6][2] and \
            lmList[12][2] < lmList[11][2] < lmList[10][2] and \
            max(lmList[5][2], lmList[9][2]) < min(lmList[20][2], lmList[16][2], lmList[4][2]):
        count = 2
        c = 0x02
    elif lmList[20][2] < lmList[18][2] and lmList[16][2] < lmList[14][2] and \
            lmList[12][2] < lmList[10][2] and lmList[8][2] < lmList[6][2] and \
            lmList[20][1] < lmList[16][1] < lmList[12][1] < lmList[8][1] and \
            lmList[4][2] > max(lmList[17][2], lmList[13][2], lmList[9][2], lmList[5][2]):
        count = 4
        c = 0x04
    elif lmList[20][2] < lmList[18][2] and lmList[16][2] < lmList[14][2] and \
            lmList[12][2] < lmList[10][2] and lmList[8][2] < lmList[6][2] and lmList[4][2] < \
            lmList[2][2] and \
            lmList[20][1] < lmList[16][1] < lmList[12][1] < lmList[8][1] < lmList[4][1]:
        count = 5
        c = 0x05
    elif lmList[4][1] < lmList[8][1] and max(lmList[4][2], lmList[8][2]) < \
            min(lmList[20][2], lmList[16][2], lmList[12][2]) and max(lmList[3][2], lmList[7][2]) < lmList[5][2]:
        count = 9
        c = 0x09

    if draw and count != 0:
        cv2.putText(img, str(count), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return count


def judgeMoving(img, lmList, pList, draw=True):
    global lcount, rcount
    count = False
    moving = '0'
    if lmList[8][2] > lmList[5][2] and lmList[12][2] > lmList[9][2] and lmList[16][2] > lmList[13][2] and \
            lmList[20][2] > lmList[17][2] and lmList[20][1] < lmList[16][1] < lmList[12][1] and \
            lmList[12][2] > lmList[0][2]:
        count = True

    if count:
        if lmList[8][1] - pList[8][1] < -10 and abs(lmList[8][2] - pList[8][2]) < 80:  # turn right
            rcount = rcount + 1
            if rcount > 8:
                moving = 'right'
                lcount = 0
        elif lmList[8][1] - pList[8][1] > 10 and abs(lmList[8][2] - pList[8][2]) < 20:  # turn left
            lcount = lcount + 1
            if lcount > 8:
                moving = 'left'
                rcount = 0
        if draw and moving != '0':
            cv2.putText(img, moving, (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 77, 0), 2)

    return moving


open_ser()
cap = cv2.VideoCapture(0)
ser.write(chr(0x00).encode("utf-8"))
while True:
    ret, img = cap.read()
    if ret:
        img = findHands(img)
        lmList = findPosition(img)
        if len(lmList) > 20:
            count = judgeHands(lmList)
            if count != 0 and flag:
                ser.write(chr(c).encode("utf-8"))
            if len(pList) > 20:
                move = judgeMoving(img, lmList, pList)
                if flag and (move == 'right'):
                    ser.write(chr(0x10).encode("utf-8"))  # 0x10——right
                if flag and (move == 'left'):
                    ser.write(chr(0x20).encode("utf-8"))  # 0x20——left
        pList = lmList
        cv2.imshow('img', img)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
ser.close()
