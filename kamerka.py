import cv2
import os
import mediapipe as mp
import time
import pyautogui
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
sciezka_modelu = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
opcje = vision.HandLandmarkerOptions(
    
    base_options=python.BaseOptions(model_asset_path=sciezka_modelu),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)
detektor= vision.HandLandmarker.create_from_options(opcje)


kamerka = cv2.VideoCapture(2, cv2.CAP_DSHOW)
czy_kliknieto=False
ekran_szer, ekran_wys = pyautogui.size()
pyautogui.PAUSE = 0
margines=100
while True:
    lewa=None
    prawa=None
    sukces, klatka=kamerka.read()
    klatka=cv2.flip(klatka, 1)
    klatkargb=cv2.cvtColor(klatka, cv2.COLOR_BGR2RGB)
    mp_obraz=mp.Image(image_format=mp.ImageFormat.SRGB, data=klatkargb)
    czas_ms=int(time.time()*1000)
    wyniki=detektor.detect_for_video(mp_obraz, czas_ms)
    wys, szer, _ = klatka.shape
    cv2.rectangle(klatka, (margines, margines), (szer - margines, wys - margines), (255, 0, 255), 2)
    if wyniki.hand_landmarks:
        for i,dlon in enumerate(wyniki.hand_landmarks):
            strona=wyniki.handedness[i][0].category_name
            if strona=="Left":
                lewa=dlon
            elif strona=="Right":
                prawa=dlon
            for punkt in dlon:
                px=int(punkt.x*szer)
                py=int(punkt.y*wys)
                cv2.circle(klatka, (px, py), 5, (255,255,0),-1)
    if lewa:
        x4l=int(lewa[4].x*szer)
        y4l=int(lewa[4].y*wys)
        x8l=int(lewa[8].x*szer)
        y8l=int(lewa[8].y*wys)
        cv2.line(klatka,(x4l,y4l),(x8l,y8l),(255,0,0),3)
        if y8l<y4l:
                cv2.circle(klatka, (x4l,y4l),12,(0,255,9), -1)
                ulamek_x = (x8l - margines) / (szer - 2 * margines)
                ulamek_y = (y8l - margines) / (wys - 2 * margines)

                kurs_x = int(ulamek_x * ekran_szer)
                kurs_y = int(ulamek_y * ekran_wys)

                kurs_x = max(1, min(ekran_szer - 1, kurs_x))
                kurs_y = max(1, min(ekran_wys - 1, kurs_y))

                pyautogui.moveTo(kurs_x, kurs_y)
    if prawa:
        x4p=int(prawa[4].x*szer)
        y4p=int(prawa[4].y*wys)
        x8p=int(prawa[8].x*szer)
        y8p=int(prawa[8].y*wys)
        x12p=int(prawa[12].x*szer)
        y12p=int(prawa[12].y*wys)
        x16p=int(prawa[16].x*szer)
        y16p=int(prawa[16].y*wys)
        if abs(x8p-x4p) < 35 and abs(y8p-y4p) < 35:
            if not czy_kliknieto:
                pyautogui.click()
                czy_kliknieto=True
        else:
            czy_kliknieto=False
        if abs(y12p-y16p) > 50 and y12p>y16p:
            pyautogui.scroll(100)
        if abs(y12p-y16p) > 50 and y16p>y12p:
            pyautogui.scroll(-100)
    cv2.imshow("Kamerka", klatka)
    if cv2.waitKey(1) == ord('q'): 
        break
detektor.close()
kamerka.release()
cv2.destroyAllWindows()