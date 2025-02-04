#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 10:26:18 2024

@author: telescopelight
"""

import cv2 as cv
import numpy as np
import pickle

def stereo_calibrate(left_path, right_path, check_size, frame_size, sample=200):
    caps = [cv.VideoCapture(left_path), cv.VideoCapture(right_path)]
    # 영상 열기 예외 처리 추가
    if not all(cap.isOpened() for cap in caps):
        print("Error: 비디오 파일을 열 수 없습니다.")
        return None

    obj_points = []    
    img_points_L = []
    img_points_R = []
    
    objp = np.zeros((np.prod(check_size), 3), np.float32)
    objp[:, :2] = np.mgrid[0:check_size[0], 0:check_size[1]].T.reshape(-1, 2)
    
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # FPS 계산 및 프레임 추출 간격 설정 (초당 2프레임)
    fps_L = int(caps[0].get(cv.CAP_PROP_FPS))
    fps_R = int(caps[1].get(cv.CAP_PROP_FPS))
    average_interval = max(1, min(fps_L, fps_R) // 5)  # 최소 1프레임 간격 보장

    frame_count = 0
    
    while all(cap.isOpened() for cap in caps) and len(obj_points) < sample:
        ret_L, frame_L = caps[0].read()
        ret_R, frame_R = caps[1].read()
        if not ret_L or not ret_R:
            break
        
        # 프레임 동기화 검사 추가
        if frame_count % average_interval == 0:
            gray_L = cv.cvtColor(frame_L, cv.COLOR_BGR2GRAY)
            gray_R = cv.cvtColor(frame_R, cv.COLOR_BGR2GRAY)
            
            ret_L, corner_L = cv.findChessboardCorners(gray_L, check_size, None)
            ret_R, corner_R = cv.findChessboardCorners(gray_R, check_size, None)
            
            if ret_L and ret_R:
                # objp 복사본 사용으로 메모리 안정성 향상
                obj_points.append(objp.copy())
                
                # 서브픽셀 정밀화
                good_corner_L = cv.cornerSubPix(gray_L, corner_L, (11, 11), (-1, -1), criteria)
                good_corner_R = cv.cornerSubPix(gray_R, corner_R, (11, 11), (-1, -1), criteria)
                
                img_points_L.append(good_corner_L)
                img_points_R.append(good_corner_R)
                print(f"{len(obj_points)} 프레임 스테레오 코너 추출 완료")
        
        frame_count += 1
    
    # 리소스 정리
    for cap in caps:
        cap.release()
    
    # 최소 프레임 수 검증
    if len(obj_points) < 20:  # OpenCV 권장 최소 샘플 수
        print(f"Error: 충분한 프레임 수집 실패 ({len(obj_points)}/{sample})")
        return None

    # 캘리브레이션 데이터 로드 (예외 처리 추가)
    try:
        with open('./output/left_calib_data.pkl', 'rb') as f:
            left_calib_data = pickle.load(f)
        with open('./output/right_calib_data.pkl', 'rb') as f:
            right_calib_data = pickle.load(f)
    except FileNotFoundError:
        print("Error: 단일 카메라 캘리브레이션 파일 누락")
        return None

    mtxL, distL = left_calib_data['camera_matrix'], left_calib_data['dist']
    mtxR, distR = right_calib_data['camera_matrix'], right_calib_data['dist']

    # 실제 프레임 크기 사용으로 정확도 향상
    actual_frame_size = (int(caps[0].get(cv.CAP_PROP_FRAME_WIDTH)), 
                        int(caps[0].get(cv.CAP_PROP_FRAME_HEIGHT)))

    # 스테레오 캘리브레이션 수행
    criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-5)
    retStereo, _, _, _, _, R, T, _, _ = cv.stereoCalibrate(
        obj_points, img_points_L, img_points_R, 
        mtxL, distL, mtxR, distR, 
        actual_frame_size,  # 실제 프레임 크기 사용
        criteria=criteria_stereo,
        flags=cv.CALIB_FIX_INTRINSIC
    )

    # 결과 유효성 검사
    if retStereo > 1.0:  # RMS 오차 임계값
        print(f"Warning: 높은 재투영 오차 ({retStereo:.2f} pixels)")

    stereo_calib_data = {'R': R, 'T': T}
    return stereo_calib_data