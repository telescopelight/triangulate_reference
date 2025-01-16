#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 18:14:41 2024

@author: telescopelight
"""

import cv2
import numpy as np
import os
import csv


def extract_white_pixel_x_values(video_path, csv_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)  # 흰색 픽셀 추출

        white_pixels = np.where(thresh == 255)
        if white_pixels[1].size > 0:
            min_x = np.min(white_pixels[1])
            max_x = np.max(white_pixels[1])
        else:
            min_x = None
            max_x = None

        data.append([min_x, max_x])

    cap.release()
    print(data)
    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Frame', 'Min_X', 'Max_X'])
        for i, (min_x, max_x) in enumerate(data):
            csvwriter.writerow([i, min_x, max_x])

if __name__ == "__main__":
    # 입력 및 출력 경로 설정

    reversed_video_path = 'reversed.mp4'
    csv_output_path = 'white_pixel_x_values.csv'


    # 흰색 픽셀 x값의 최솟값과 최대값 추출 및 저장
    extract_white_pixel_x_values(reversed_video_path, csv_output_path)
    print(f"White pixel x values saved to {csv_output_path}")