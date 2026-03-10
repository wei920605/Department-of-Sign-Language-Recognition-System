# Department-of-Sign-Language-Recognition-System

A real-time sign language recognition system using Python, OpenCV and MediaPipe.

## Features
- Real-time hand detection
- Sign language recognition
- Text and speech output

## Tech Stack
- Python
- OpenCV
- MediaPipe
- TensorFlow

## How to run
pip install -r requirements.txt
python main.py
檔案說明
需先錄製一段影片放入檔案中
再跑這段程式碼`back.py`: (影片去背且以每秒提取 5 幀的邏輯穩定執行)
      圖片數=每秒提取幀數×影片長度(秒)
      ex.影片長度為 5 秒->圖片數 5×5=25(張)
      確保獲得足夠的動作信息，同時控制數據量，避免過度負擔計算資源。
      
再來跑這段`change.py`: (資料擴增:水平翻轉（Horizontal Flip)、旋轉（Rotate）、色彩抖動（Color Jitter）)
 
`main.py`: 主程式
`keras_model.h5`: 訓練好的手語辨識模型
`labels.txt`: 模型對應的標籤類別
