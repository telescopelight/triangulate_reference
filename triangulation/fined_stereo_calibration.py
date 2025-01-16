#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 10:26:18 2024

@author: telescopelight
"""

import cv2 as cv
import numpy as np
import pickle

def stereo_calibrate(left_path, right_path, check_size, frame_size, sample=500):
    caps = [cv.VideoCapture(left_path), cv.VideoCapture(right_path)]  # 영상 행렬화
    obj_points = []
    
    img_points_L = []
    img_points_R = []
    
    objp = np.zeros((np.prod(check_size), 3), np.float32)
    objp[:, :2] = np.mgrid[0:check_size[0], 0:check_size[1]].T.reshape(-1, 2)
    
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # FPS 읽기
    fps_L = int(caps[0].get(cv.CAP_PROP_FPS))
    fps_R = int(caps[1].get(cv.CAP_PROP_FPS))
    average_interval = min(fps_L, fps_R) // 2  # 초당 2개의 프레임을 추출하기 위한 간격 계산

    frame_count = 0
    
    while caps[0].isOpened() and caps[1].isOpened() and len(obj_points) < sample:
        ret_L, frame_L = caps[0].read()
        ret_R, frame_R = caps[1].read()
        if not ret_L or not ret_R:
            break
        
        if frame_count % average_interval == 0:
            gray_L = cv.cvtColor(frame_L, cv.COLOR_BGR2GRAY)
            gray_R = cv.cvtColor(frame_R, cv.COLOR_BGR2GRAY)
            
            ret_L, corner_L = cv.findChessboardCorners(gray_L, check_size, None)
            ret_R, corner_R = cv.findChessboardCorners(gray_R, check_size, None)
            
            if ret_L and ret_R:
                obj_points.append(objp)
                
                good_corner_L = cv.cornerSubPix(gray_L, corner_L, (11, 11), (-1, -1), criteria)
                good_corner_R = cv.cornerSubPix(gray_R, corner_R, (11, 11), (-1, -1), criteria)
                
                img_points_L.append(good_corner_L)
                img_points_R.append(good_corner_R)  # 좋은 서브 픽셀 추가
                print(f"{len(obj_points)} 프레임 스테레오 코너 추출 완료")
        
        frame_count += 1
            
    caps[0].release()
    caps[1].release()
    # 서브픽셀 정밀화까지 완료
    
    if len(obj_points) < sample:  # 샘플보다 적으면
        print(f"추출된 프레임 갯수: {len(obj_points)}, 목표 추출 프레임: {sample}")
        
    with open('./output/left_calib_data.pkl', 'rb') as f:
        left_calib_data = pickle.load(f)
    with open('./output/right_calib_data.pkl', 'rb') as f:
        right_calib_data = pickle.load(f)
        
    mtxL, distL, _, _ = left_calib_data.values()  # 데이터 불러오기
    mtxR, distR, _, _ = right_calib_data.values()
    
    flag = 0
    flag |= cv.CALIB_FIX_INTRINSIC  # 내부 카메라 매개변수 고정 비트 or= 연산자, 특이함
    
    criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-5)
    retStereo, _, _, _, _, R, T, _, _ = cv.stereoCalibrate(
        obj_points, img_points_L, img_points_R, mtxL, distL, mtxR, distR, frame_size, criteria_stereo, flag
    )

    stereo_calib_data = {
        'R': R,
        'T': T
    }
    
    print(f"Stereo Calibration completed with RMS error: {retStereo}")
    print(f"Rotation matrix:\n{R}")
    print(f"Translation vector:\n{T}")
    
    return stereo_calib_data
