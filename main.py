import cv2
face_ref = cv2.CascadeClassifier("face_ref.xml") # Referensi wajah
camera = cv2.VideoCapture(0)

def face_detection(frame):
    optimized_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_ref.detectMultiScale(optimized_frame, minSize=(50,50), scaleFactor=1.1, minNeighbors=4);
    return faces;

def drawer_box(frame):
    for x, y, w, h in face_detection(frame):
        cv2.rectangle(frame, (x, y),(x + w, x + h), (255,0,000), 4);
    pass

def close_window():
    camera.release()
    cv2.destroyAllWindows()
    exit()

def main():
    while True:
        _, frame = camera.read()
        drawer_box(frame)
        cv2.imshow("AI Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            close_window()

if __name__ == '__main__':
    main()
