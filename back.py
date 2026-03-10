#影片去背
import cv2
import numpy as np
import mediapipe as mp
import os

mp_selfie_segmentation = mp.solutions.selfie_segmentation

# 輸入影片的檔案清單
video_files = ['rain.mp4','left_rain.mp4','right_rain.mp4','flipped_rain.mp4','jittered_rain.mp4',
               'rain1.mp4','left_rain1.mp4','right_rain1.mp4','flipped_rain1.mp4','jittered_rain1.mp4',
               'rain2.mp4','left_rain2.mp4','right_rain2.mp4','flipped_rain2.mp4','jittered_rain2.mp4',
               'rain3.mp4','left_rain3.mp4','right_rain3.mp4','flipped_rain3.mp4','jittered_rain3.mp4',
               'rain4.mp4','left_rain4.mp4','right_rain4.mp4','flipped_rain4.mp4','jittered_rain4.mp4',
               'rain5.mp4','left_rain5.mp4','right_rain5.mp4','flipped_rain5.mp4','jittered_rain5.mp4',
               'rain6.mp4','left_rain6.mp4','right_rain6.mp4','flipped_rain6.mp4','jittered_rain6.mp4',
               'rain7.mp4','left_rain7.mp4','right_rain7.mp4','flipped_rain7.mp4','jittered_rain7.mp4']

# 輸出圖片的資料夾
output_folder = 'output_images'

# 如果輸出資料夾不存在，則建立它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for video_file in video_files:
    # Initialize the video capture object
    cap = cv2.VideoCapture(video_file)

    # Create a background subtractor object
    backSub = cv2.createBackgroundSubtractorMOG2()

    frame_idx = 0  # 幀計數器
    fps = cap.get(cv2.CAP_PROP_FPS)  # 獲取影片的每秒幀數 (FPS)
    frame_interval = int(round(fps / 5))  # 每秒提取 5 幀

    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        if frame_idx % frame_interval == 0: # 每隔 n 幀處理一次
            # 將影像的色彩空間從 BGR 轉換為 RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 使用 MediaPipe Selfie Segmentation 處理影像
            results = selfie_segmentation.process(image_rgb)

            # 獲取分割遮罩（segmentation mask）
            segmentation_mask = results.segmentation_mask

            # 將遮罩轉換為二值影像
            segmentation_mask = (segmentation_mask > 0.5).astype(np.uint8) * 255

            # 創建純黑背景的影像
            background = np.zeros_like(frame)

            # 將前景（人物）和背景結合
            output_image = np.where(segmentation_mask[:,:,None], frame, background)

            # Save the result as an image
            output_file = os.path.join(output_folder, f'{os.path.splitext(os.path.basename(video_file))[0]}_{frame_idx:04d}.png')
            cv2.imwrite(output_file, output_image)

        frame_idx += 1

    # Release the video capture object
    cap.release()
