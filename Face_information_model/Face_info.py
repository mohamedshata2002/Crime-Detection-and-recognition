from  Face_information_model import f_Face_info
import cv2
import time
import imutils
import os

# def face_image(frame):
#     # frame = cv2.imread(input_path)
#     out = f_Face_info.get_face_info(frame)
#     res_img = f_Face_info.bounding_box(out, frame)
#     # Generate output filename
#     # filename = os.path.basename(input_path)
#     # output_path = os.path.join(output_folder, 'processed_' + filename)
#     # cv2.imwrite(output_path, res_img)
#     return res_img  ,out
    

# # output_folder = r'E:\workstation\\project\\project\\crime_detection\\Face_information_model\\New folder\\output'

# # input_path = r'E:\workstation\\project\\project\\crime_detection\\Face_information_model\\New folder\\input\\img.jpg'
# # # output_folder = r'D:\Face_information_model\results'
# # os.makedirs(output_folder, exist_ok=True)
# # process_image(input_path, output_folder)
# # # # input_folder = r'D:\Face_information_model\data_test'
# # # execution_times = []

# # for filename in os.listdir(input_folder):
# #     if filename.endswith(".jpg") or filename.endswith(".png"):
# #         input_path = os.path.join(input_folder, filename)

# #         start_time = time.time()
# #         process_image(input_path, output_folder)
# #         end_time = time.time()

# #         execution_time = end_time - start_time
# #         execution_times.append(execution_time)

# #         print(f"Processed {filename} in {execution_time:.2f} seconds.")

# # average_execution_time = sum(execution_times) / len(execution_times)
# # print(f"Average execution time: {average_execution_time:.2f} seconds.")


def face_video(input_path,output_path):
    # Open the input video
    
    video_capture = cv2.VideoCapture(input_path)
    # Get the video properties
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    # Create VideoWriter object to save the output video
    
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    print(output_path)
    start_time = time.time()
    genders = []
    names = []
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Process the frame using the Face_info module
        out = f_Face_info.get_face_info(frame)
        res_img ,gender,name= f_Face_info.bounding_box(out, frame)
        genders.append(gender)
        names.append(name)
        video_writer.write(res_img)

    end_time = time.time()
    execution_time = end_time - start_time
    genders = [item for items in genders for item in items]
    names = [item for items in names for item in items]
    face_data = list(zip(names, genders))
    face_data = set(face_data)
    video_capture.release()
    video_writer.release()
    return face_data



# output_folder = r'E:\workstation\\project\\project\\crime_detection\\Face_information_model\\New folder\\output'

# input_folder = r'E:\workstation\\project\\project\\crime_detection\\Face_information_model\\New folder\\input\\test.mp4'
# os.makedirs(output_folder, exist_ok=True)

# process_video(input_folder, output_folder)