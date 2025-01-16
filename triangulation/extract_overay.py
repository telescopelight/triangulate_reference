#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 12:38:59 2024

@author: telescopelight
"""

import cv2
import numpy as np

def extract_and_overlay_edges(input_video_path, output_video_path, top_fraction=0.43, fade_duration=60):
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps/5, (width, height))
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        return
    
    height_top = int(height * top_fraction)
    height_low = int(height * (1 - top_fraction))
    
    edge_frames = []

    while ret:
        # 상단 영역만 추출
        top_area = frame[height_top:height_low, :]
        
        # 그레이스케일 변환 및 엣지 검출
        gray = cv2.cvtColor(top_area, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 100)
        
        # 빨간색 엣지 프레임 생성
        red_edges = np.zeros_like(frame)
        red_edges[height_top:height_low, :][edges != 0] = [0, 0, 255]

        # 엣지 프레임 스택에 추가
        edge_frames.append(red_edges)
        
        # 이전 엣지 프레임들에 잔상 효과 적용
        faded_edges = np.zeros_like(frame)
        for i in range(len(edge_frames)):
            alpha = max(0, 1 - (len(edge_frames) - 1 - i) / fade_duration)
            faded_edges = cv2.addWeighted(faded_edges, 1, edge_frames[i], alpha, 0)
        
        # 너무 많은 프레임이 저장되지 않도록 함
        if len(edge_frames) > fade_duration:
            edge_frames.pop(0)
        
        # 잔상이 적용된 엣지 프레임을 원본 프레임에 합성
        combined_frame = cv2.addWeighted(frame, 1, faded_edges, 1, 0)
        
        # 결과 프레임을 동영상에 저장
        out.write(combined_frame)
        
        # 다음 프레임 읽기
        ret, frame = cap.read()
    
    cap.release()
    out.release()
    print("Video processing completed and saved.")

# 실행 예제
extract_and_overlay_edges('detect.mp4', 'output_video_with_edges.mp4')