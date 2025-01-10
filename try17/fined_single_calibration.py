#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 10:13:06 2024

@author: telescopelight

초당 2개의 프레임만 추출해 캘리브레이션에 이용하는 코드
"""

import cv2
import numpy as np
import pickle

def calibrate_camera(input_path, chessboard_size = (9,6), frame_size = (1920,1080), num_frames_to_calibrate=20):
    cap = cv2.VideoCapture(input_path)
    objpoints = []  # 3D 점
    imgpoints = []  # 2D 점
    
    # 비교 대상이 될 현실 포인트 좌표계
    objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # FPS와 총 프레임 수 읽기
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_interval = fps // 2  # 초당 2개의 프레임을 추출하기 위한 간격
    
    print(f"FPS: {fps}, Total Frames: {total_frames}, Frame Interval: {frame_interval}")

    frame_count = 0
    extracted_frames = 0

    while cap.isOpened() and len(objpoints) < num_frames_to_calibrate:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                extracted_frames += 1
                print(f"Frame {frame_count}: Chessboard corners detected and refined.")
        
        frame_count += 1

    cap.release()
    
    if len(objpoints) < num_frames_to_calibrate:
        print(f"Warning: Only {len(objpoints)} valid frames found for calibration.")
    
    ret, camera_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frame_size, None, None)

    calib_data = {
        'camera_matrix': camera_matrix,
        'dist': dist,
        'rvecs': rvecs,
        'tvecs': tvecs
    }  # 보정 데이터 얻어
    
    return calib_data
"""
chessboard_size = (9, 6)
frame_size = (1920, 1080)
num_frames_to_calibrate = 50

left_calib_data = calibrate_camera('left_cali.mp4', chessboard_size, frame_size, num_frames_to_calibrate)
right_calib_data = calibrate_camera('right_cali.mp4', chessboard_size, frame_size, num_frames_to_calibrate)

with open('left_calib_data.pkl', 'wb') as f:
    pickle.dump(left_calib_data, f)
with open('right_calib_data.pkl', 'wb') as f:
    pickle.dump(right_calib_data, f)"""