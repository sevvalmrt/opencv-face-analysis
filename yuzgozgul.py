import cv2
import os

# Haarcascade dosyalarının bulunduğu klasörü belirle
cascade_path = os.path.join(os.getcwd(), "haarcascade")

# Modelleri yükle
yuzCascade = cv2.CascadeClassifier(os.path.join(cascade_path, "haarcascade_frontalface_default.xml"))
eyeCascade = cv2.CascadeClassifier(os.path.join(cascade_path, "haarcascade_eye.xml"))
smileCascade = cv2.CascadeClassifier(os.path.join(cascade_path, "haarcascade_smile.xml"))

# Kamerayı başlat
kamera = cv2.VideoCapture(0)
kamera.set(3, 1280)  # Genişlik
kamera.set(4, 720)   # Yükseklik

while True:
    ret, kare = kamera.read()
    if not ret:
        print("Kamera görüntüsü alınamadı!")
        break
    
    gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
    
    # Yüz tespiti
    yuzler = yuzCascade.detectMultiScale(gri, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
    
    for (x, y, w, h) in yuzler:
        # Yüzü elipsle işaretle
        cv2.ellipse(kare, (x + w // 2, y + h // 2), (w // 2, h // 2), 5, 0, 360, (255, 0, 0), 2)
        
        gri_kutu = gri[y:y + h, x:x + w]
        renkli_kutu = kare[y:y + h, x:x + w]
        
        # Göz tespiti
        gozler = eyeCascade.detectMultiScale(gri_kutu, scaleFactor=1.05, minNeighbors=5, minSize=(40, 40))
        for (ex, ey, ew, eh) in gozler:
            cv2.rectangle(renkli_kutu, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Gülümseme tespiti
        gulusler = smileCascade.detectMultiScale(gri_kutu, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))
        for (sx, sy, sw, sh) in gulusler:
            cv2.rectangle(renkli_kutu, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

    cv2.imshow("Yüz, Göz ve Gülümseme Tespiti", kare)
    
    # Çıkış için ESC veya 'q' tuşuna bas
    k = cv2.waitKey(10) & 0xFF
    if k == 27 or k == ord("q"):
        break

kamera.release()
cv2.destroyAllWindows()
