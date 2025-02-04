
import cv2
import os

# Haarcascade dosyasını proje klasöründen yükle
cascade_path = os.path.join(os.getcwd(), "haarcascade_frontalface_default.xml")
yuzCascade = cv2.CascadeClassifier(cascade_path)

# Yeni bir pencere oluştur (otomatik boyut)
cv2.namedWindow("Yüz Algılama", cv2.WINDOW_AUTOSIZE)

# Kamerayı başlat
kamera = cv2.VideoCapture(0)
if not kamera.isOpened():
    print("Hata: Kamera açılamadı!")
    exit()

# Kameranın ters olup olmadığını belirleyen değişken
ters_cevir = False

while True:
    _, kare = kamera.read()
    if ters_cevir:
        kare = cv2.flip(kare, -1)

    gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
    yuzler = yuzCascade.detectMultiScale(
        gri,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )

    for (x, y, w, h) in yuzler:
        cv2.rectangle(kare, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Yüz Algılama", kare)

    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord("q"):  # ESC veya 'q' basıldığında çık
        break

kamera.release()
cv2.destroyAllWindows()
