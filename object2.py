import cv2
import numpy as np
import webbrowser

# Global variables to store the clicked object
clicked_object = None
clicked_label = None
boxes = []  # To store bounding boxes
class_ids = []  # To store class IDs

# Mouse callback function
def click_event(event, x, y, flags, param):
    global clicked_object, clicked_label
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, box in enumerate(boxes):
            x1, y1, w, h = box
            if x1 < x < x1 + w and y1 < y < y1 + h:  # Check if the click is within the bounding box
                clicked_object = box
                clicked_label = classes[class_ids[i]]
                print(f"Clicked on: {clicked_label}")
                break

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers().flatten()
output_layers = [layer_names[i - 1] for i in output_layers_indices]

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    cv2.namedWindow("Camera Feed")
    cv2.setMouseCallback("Camera Feed", click_event)  # Set mouse callback function

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Prepare the frame for YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        # Initialize lists for detected objects
        class_ids = []
        confidences = []
        boxes = []

        # Process the outputs
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Confidence threshold
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Maxima Suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Draw bounding boxes and labels
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        # Check if an object was clicked
        if clicked_object is not None and clicked_label is not None:
            # Open Google search for the clicked object
            webbrowser.open(f"https://www.google.com/search?q={clicked_label}", new=2)
            clicked_object = None  # Reset clicked object
            clicked_label = None  # Reset clicked label

        # Display the resulting frame
        cv2.imshow("Camera Feed", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()