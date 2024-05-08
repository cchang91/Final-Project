import cv2
from inference.models.utils import get_roboflow_model

# Roboflow model
model_name = "comphoto"
model_version = "5"

# Get Roboflow model
model = get_roboflow_model(
    model_id="{}/{}".format(model_name, model_version),
    api_key="GE5hI6Y9Set5IYtYPU13"
)

# Open the default camera (usually the built-in webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If the frame was read successfully, display it
    if ret:
        # Run inference on the frame
        results = model.infer(image=frame, confidence=0.1, iou_threshold=0.1)

        # Check if results contain any detections
        if results and 'predictions' in results:
            # Get predictions
            predictions = results['predictions']

            # Iterate over predictions
            for pred in predictions:
                # Extract bounding box coordinates and class
                x, y, width, height = pred['x'], pred['y'], pred['width'], pred['height']
                label = pred['class']
                confidence = pred['confidence']

                # Calculate corner points
                x1, y1 = int(x - width / 2), int(y - height / 2)
                x2, y2 = int(x + width / 2), int(y + height / 2)

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Webcam Feed', frame)

        # Press 'q' to quit the video window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Error: Could not read frame.")
        break

# When everything is done, release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
