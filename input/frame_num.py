#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 2024

동영상의 각 프레임에 프레임 번호를 붙여 저장하는 코드
"""

import cv2

def add_frame_number_overlay(input_video_path, output_video_path, text_position=(50, 50)):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {input_video_path}")
        return

    # 입력 동영상의 속성 읽기
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video properties: {width}x{height}, FPS: {fps}, Total frames: {frame_count}")

    # 동영상 저장을 위한 VideoWriter 생성
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 번호를 텍스트로 생성 (원하는 형식 및 위치로 조정 가능)
        text = f"Frame: {frame_idx}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 255, 0)  # 초록색 텍스트
        thickness = 2
        cv2.putText(frame, text, text_position, font, font_scale, color, thickness, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Processing complete. Output saved to {output_video_path}")

if __name__ == '__main__':
    # 입력 동영상 파일 경로와 저장할 동영상 파일 경로 (적절히 수정하세요)
    input_video_path = './input/raw_L.mp4'
    output_video_path = './input/num_raw_L.mp4'
    add_frame_number_overlay(input_video_path, output_video_path)