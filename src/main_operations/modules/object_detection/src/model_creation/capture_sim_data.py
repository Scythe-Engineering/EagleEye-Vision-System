import cv2

# Stream URL
url = "http://10.0.0.62:1181/1?fps=60"

# Open the video stream
cap = cv2.VideoCapture(url)

# Check if the stream opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Get video frame width, height, and FPS
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30  # Default to 30 FPS if unknown

# Define the codec and create VideoWriter object (MP4 with H.264)
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

print("Recording... Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Write the frame to file
    out.write(frame)

    # Display the video (optional)
    cv2.imshow("Stream", frame)

    # Press 'q' to quit recording
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Recording stopped and saved as output.mp4")
