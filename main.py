# Face Images Recognition
import cv2
import face_recognition

# Face Camera Recognition
#from simple_fc import SimpleFacerec

if __name__ == '__main__':

    # Camara
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        cv2.imshow("Frame", frame)

        # Capturar el cuadro del video
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Cargar las imagenes

    # Ellon Musk
    img = cv2.imread("./faces/ellon.png")
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_encoding = face_recognition.face_encodings(rgb_img)[0]

    # Jeff Bezzos
    img2 = cv2.imread("./faces/jeff.png")
    rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

    # Ellon Two
    img3 = cv2.imread("./faces/ellon2.png")
    rgb_img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    img_encoding3 = face_recognition.face_encodings(rgb_img3)[0]

    # Jeff Two
    img4 = cv2.imread("./faces/ellon2.png")
    rgb_img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
    img_encoding4 = face_recognition.face_encodings(rgb_img4)[0]

    result = face_recognition.compare_faces([img_encoding], img_encoding3)
    print("Resultado: ", result)

    cv2.imshow("Img", img)
    cv2.waitKey(0)



