
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 20:17:03 2024

@author: telescopelight
"""

import cv2

def cut_video(input_file, output_file, start_time, end_time):
    # 비디오 파일 열기
    cap = cv2.VideoCapture(input_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #총 프레임 가져오기
    
    # 시작 프레임과 종료 프레임 계산
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    # 비디오 라이터 객체 생성
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 선택
    out = cv2.VideoWriter(output_file, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
    
    current_frame = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if current_frame >= start_frame and current_frame <= end_frame:
            out.write(frame)
        
        current_frame += 1
        if current_frame > end_frame:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# 예제 사용
#input_file = 'right.mp4'
#output_file = 'c_right.mp4'
start_time =0  # 시작 시간 (초)
end_time = 20   # 종료 시간 (초)


cut_video('left_calib.mp4', 'cut_cali_left.mp4', 52/60, 52/60+5)
cut_video('right_calib.mp4', 'cut_cali_right.mp4', 13/60, 13/60+5)
