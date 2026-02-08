import cv2
from fer import FER

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize FER detector without MTCNN
detector = FER(mtcnn=False)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect emotions in the frame
    results = detector.detect_emotions(frame)

    for face in results:
        (x, y, w, h) = face["box"]
        # Get dominant emotion
        emotion = max(face["emotions"], key=face["emotions"].get)
        
        # Draw bounding box and label (only emotion)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Live Emotion Detection", frame)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
