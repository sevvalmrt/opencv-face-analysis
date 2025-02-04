import cv2


# Haarcascade dosyalarını yükle
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# Kamerayı başlat
kamera = cv2.VideoCapture(0)
kamera.set(3, 1280)  # Genişlik
kamera.set(4, 720)   # Yükseklik

dosya_ad = None  # 'tespit_kaydi.mp4'
kaydedici = None

while True:
    ret, kare = kamera.read()
    if not ret:
        print("Hata: Kamera görüntüsü alınamadı!")
        break

    gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
    yuzler = face_cascade.detectMultiScale(gri, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in yuzler:
        cv2.rectangle(kare, (x, y), (x + w, y + h), (255, 0, 0), 2)
        gri_yuz = gri[y:y + h, x:x + w]
        renkli_yuz = kare[y:y + h, x:x + w]

        # Gözleri algıla
        gozler = eye_cascade.detectMultiScale(gri_yuz, scaleFactor=1.5, minNeighbors=10, minSize=(3, 3))
        for (ex, ey, ew, eh) in gozler:
            cv2.rectangle(renkli_yuz, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        
        # Gülümsemeyi algıla
        gulusler = smile_cascade.detectMultiScale(gri_yuz, scaleFactor=1.5, minNeighbors=18, minSize=(30, 30))
        for (sx, sy, sw, sh) in gulusler:
            cv2.rectangle(renkli_yuz, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

    cv2.imshow("Yüz, Göz ve Gülümseme Algılama", kare)
    
    # Video kaydı başlat
    if kaydedici is None and dosya_ad is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        kaydedici = cv2.VideoWriter(dosya_ad, fourcc, 24.0, (kare.shape[1], kare.shape[0]), True)
    
    if kaydedici is not None:
        kaydedici.write(kare)
    
    # Çıkış için ESC veya 'q' tuşu
    k = cv2.waitKey(10) & 0xFF
    if k == 27 or k == ord('q'):
        break

kamera.release()
if kaydedici:
    kaydedici.release()
cv2.destroyAllWindows()
