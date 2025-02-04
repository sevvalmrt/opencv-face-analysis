import cv2

def yukle_model(yuz_path, goz_path):
    yuzCascade = cv2.CascadeClassifier(yuz_path)
    gozCascade = cv2.CascadeClassifier(goz_path)
    return yuzCascade, gozCascade

def tespit_et(kare, yuzCascade, gozCascade):
    gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
    yuzler = yuzCascade.detectMultiScale(gri, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))

    for (x, y, w, h) in yuzler:
        cv2.rectangle(kare, (x, y), (x+w, y+h), (255, 0, 0), 2)
        gri_kutu = gri[y:y+h, x:x+w]
        renkli_kutu = kare[y:y+h, x:x+w]
        gozler = gozCascade.detectMultiScale(gri_kutu, scaleFactor=1.2, minNeighbors=10, minSize=(3, 3))

        for (ex, ey, ew, eh) in gozler:
            cv2.rectangle(renkli_kutu, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

def main():
    yuzCascade, gozCascade = yukle_model(
        r"D:\github\ilk\haarcascade_frontalface_default.xml",
        r"D:\github\iki\haarcascade_eye.xml"
    )

    kamera = cv2.VideoCapture(0)
    kamera.set(3, 1280)
    kamera.set(4, 720)

    dosyaad = None  # 'goz_saptama.mp4'
    kaydedici = None

    while True:
        _, kare = kamera.read()
        tespit_et(kare, yuzCascade, gozCascade)
        cv2.imshow('kare', kare)

        if kaydedici is None and dosyaad is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            kaydedici = cv2.VideoWriter(dosyaad, fourcc, 24.0, (kare.shape[1], kare.shape[0]), True)

        if kaydedici is not None:
            kaydedici.write(kare)

        k = cv2.waitKey(10) & 0xff
        if k == 27 or k == ord('q'):
            break

    kamera.release()
    if kaydedici:
        kaydedici.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
