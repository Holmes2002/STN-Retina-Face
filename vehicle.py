
from ultralytics import YOLO
import cv2
import os

# Load the YOLOv8 model
model = YOLO("/home1/data/vinhnguyen/Flag_detection/runs_demo_model_s_detect_7_class_v2/train/weights/best.pt")
root = "/home1/data/vinhnguyen/Deepstream/video_hawkice/Situation_1_vehical"
for index_video, video_path in enumerate(os.listdir(root)):
    index_video = ''
    video_path = 'ScreenRecording2023-09-12at14.51.55.mov'
    # Define the path to the input video
    video_path = f'{root}/{video_path}'

    # Define the frame skip interval
    frame_skip = 1

    # Create output directory if not exists
    output_dir = f"cropped_objects/{index_video}"
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    index = 0
    os.makedirs(output_dir, exist_ok=True)

    frame_count = 0

    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()
        
        if not ret:
            break  # Break if no more frames are available

        # Only process every `frame_skip` frame
        if frame_count % frame_skip == 0:
            index += 1
            
            # Run inference on the current frame
            results = model([frame])

            # Define the class indices to crop (classes 3-6: car, motorcycle, bus, truck)
            target_classes = [3, 4, 5, 6]

            # Process results list
            for idx, result in enumerate(results):
                # Original image (the current frame)
                img = result.orig_img

                # Loop through each detection
                for i, (box, cls) in enumerate(zip(result.boxes.xyxy, result.boxes.cls)):
                    # Convert tensor to numpy array
                    box = box.cpu().numpy().astype(int)
                    cls = int(cls.cpu().numpy())

                    # Check if the class is in target_classes
                    if cls in target_classes:
                        # Crop the detected object using bounding box coordinates
                        x1, y1, x2, y2 = box
                        cropped_img = img[y1:y2, x1:x2]

                        # Define the output filename
                        output_filename = os.path.join(output_dir, f"object_{index}_{idx}_{i}_class_{cls}.jpg")
                        
                        # Save the cropped image
                        cv2.imwrite(output_filename, cropped_img)
                        print(f"Cropped object saved at {output_filename}")

        frame_count += 1
    assert False

    # Release the video capture object
    cap.release()
# for index_video, video_path in enumerate(os.listdir(root)[:1]):
#     index_video = ''
#     # Define the path to the input video
#     video_path = f'{root}'

#     # Define the frame skip interval
#     frame_skip = 1
#     output_dir = f"cropped_objects"
#     index = 0
#     os.makedirs(output_dir, exist_ok=True)

#     frame_count = 0
#     for frame in os.listdir(video_path):
#             frame = cv2.imread(f'{video_path}/{frame}')
#             index += 1
            
#             # Run inference on the current frame
#             results = model([frame])

#             # Define the class indices to crop (classes 3-6: car, motorcycle, bus, truck)
#             target_classes = [3, 4, 5, 6]

#             # Process results list
#             for idx, result in enumerate(results):
#                 # Original image (the current frame)
#                 img = result.orig_img

#                 # Loop through each detection
#                 for i, (box, cls) in enumerate(zip(result.boxes.xyxy, result.boxes.cls)):
#                     # Convert tensor to numpy array
#                     box = box.cpu().numpy().astype(int)
#                     cls = int(cls.cpu().numpy())

#                     # Check if the class is in target_classes
#                     if cls in target_classes:
#                         # Crop the detected object using bounding box coordinates
#                         x1, y1, x2, y2 = box
#                         cropped_img = img[y1:y2, x1:x2]

#                         # Define the output filename
#                         output_filename = os.path.join(output_dir, f"object_{index}_{idx}_{i}_class_{cls}.jpg")
                        
#                         # Save the cropped image
#                         cv2.imwrite(output_filename, cropped_img)
#                         print(f"Cropped object saved at {output_filename}")


