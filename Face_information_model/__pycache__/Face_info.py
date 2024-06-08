import Face_information_model.f_Face_info
#import f_Face_info
import cv2
import time
import imutils
import os

def process_image(input_path, output_folder):
    frame = cv2.imread(input_path)
    out = Face_information_model.f_Face_info.get_face_info(frame)
    res_img = Face_information_model.f_Face_info.bounding_box(out, frame)

    # Generate output filename
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_folder, 'processed_' + filename)
    cv2.imwrite(output_path, res_img)

output_folder = r'C:\Users\DELL\Downloads\Face_information_model\results'
os.makedirs(output_folder, exist_ok=True)

input_folder = r'C:\Users\DELL\Downloads\Face_information_model\data_test'
execution_times = []

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        input_path = os.path.join(input_folder, filename)

        start_time = time.time()
        process_image(input_path, output_folder)
        end_time = time.time()

        execution_time = end_time - start_time
        execution_times.append(execution_time)

        print(f"Processed {filename} in {execution_time:.2f} seconds.")

average_execution_time = sum(execution_times) / len(execution_times)
print(f"Average execution time: {average_execution_time:.2f} seconds.")


def process_video(input_path):
    # Open the input video
    video_capture = cv2.VideoCapture(input_path)
    # Get the video properties
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    fps=30
    # Create VideoWriter object to save the output video
    output_folder = r'C:\Users\DELL\Downloads\crime_detection\Face_information_model\results'
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, 'processed_video.mp4')
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    start_time = time.time()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Process the frame using the Face_info module
        out = Face_information_model.f_Face_info.get_face_info(frame)
        res_img = Face_information_model.f_Face_info.bounding_box(out, frame)
        video_writer.write(res_img)

    end_time = time.time()
    execution_time = end_time - start_time
    video_capture.release()
    video_writer.release()
    print(f"Processed video in {execution_time:.2f} seconds.")

#input_video = r'C:\Users\DELL\Downloads\Face_information_model\data_test\video\Last Breath _ 10 Second Action Clip.mp4'
# output_folder = r'C:\Users\DELL\Downloads\Face_information_model\results'
# os.makedirs(output_folder, exist_ok=True)
#process_video(input_video, output_folder)