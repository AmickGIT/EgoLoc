import cv2

# Load the input video
input_path = "bathroomCabinet_10.mp4"
cap = cv2.VideoCapture(input_path)

# Define start and end frame numbers
start_frame = 127  # 79th frame (0-indexed)
end_frame = 164

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID'
out = cv2.VideoWriter('output_79_to_120.mp4', fourcc, fps, (width, height))

# Go to the start frame
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# Loop through the desired frame range
for frame_num in range(start_frame, end_frame + 1):
    ret, frame = cap.read()
    if not ret:
        print(f"Could not read frame {frame_num}. Stopping.")
        break
    out.write(frame)

# Release everything
cap.release()
out.release()
print("Output video saved as 'output_79_to_120.mp4'")
