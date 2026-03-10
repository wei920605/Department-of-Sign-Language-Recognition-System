#左右旋轉 水平翻轉 color jitter .mp4
import cv2
import numpy as np
import os

# 輸入影片的檔案清單
video_files = ['rain1.mp4', 'rain2.mp4', 'rain.mp4', 'rain3.mp4', 'rain4.mp4', 'rain5.mp4',
               'rain6.mp4', 'rain7.mp4']

# 定義左右旋轉的角度
angle_left = -15
angle_right = 15

# 定義色彩抖動的參數
brightness_range = (0.8, 1.2)
contrast_range = (0.8, 1.2)
saturation_range = (0.8, 1.2)
hue_range = (-10, 10)

def apply_color_jitter(frame):
    # 亮度調整
    brightness_factor = np.random.uniform(brightness_range[0], brightness_range[1])
    frame = cv2.convertScaleAbs(frame, alpha=brightness_factor, beta=0)

    # 對比度調整
    contrast_factor = np.random.uniform(contrast_range[0], contrast_range[1])
    frame = cv2.convertScaleAbs(frame, alpha=contrast_factor, beta=0)

    # 飽和度調整
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation_factor = np.random.uniform(saturation_range[0], saturation_range[1])
    hsv[:, :, 1] = cv2.convertScaleAbs(hsv[:, :, 1], alpha=saturation_factor, beta=0)
    
    # 色調調整
    hue_factor = np.random.randint(hue_range[0], hue_range[1])
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_factor) % 180
    
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return frame

for video_file in video_files:
    # Initialize the video capture object
    cap = cv2.VideoCapture(video_file)

    # Get the video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter objects
    output_file_left = f'left_{video_file}'
    output_file_right = f'right_{video_file}'
    output_file_flipped = f'flipped_{video_file}'
    output_file_jittered = f'jittered_{video_file}'
    
    out_left = cv2.VideoWriter(output_file_left, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    out_right = cv2.VideoWriter(output_file_right, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    out_flipped = cv2.VideoWriter(output_file_flipped, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    out_jittered = cv2.VideoWriter(output_file_jittered, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Loop through each frame in the video
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Rotate the frame to the left by 15 degrees
        M_left = cv2.getRotationMatrix2D((width / 2, height / 2), angle_left, 1)
        rotated_left = cv2.warpAffine(frame, M_left, (width, height))

        # Rotate the frame to the right by 15 degrees
        M_right = cv2.getRotationMatrix2D((width / 2, height / 2), angle_right, 1)
        rotated_right = cv2.warpAffine(frame, M_right, (width, height))

        # Flip the frame horizontally
        flipped_frame = cv2.flip(frame, 1)

        # Apply color jitter to the frame
        jittered_frame = apply_color_jitter(frame)

        # Write the rotated, flipped, and jittered frames to the output videos
        out_left.write(rotated_left)
        out_right.write(rotated_right)
        out_flipped.write(flipped_frame)
        out_jittered.write(jittered_frame)

    # Release the video capture and writer objects
    cap.release()
    out_left.release()
    out_right.release()
    out_flipped.release()
    out_jittered.release()
