#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:02:08 2024

@author: telescopelight
"""

import os
import pickle
import get_yellow_point
from fined_single_calibration import calibrate_camera
from fined_stereo_calibration import stereo_calibrate
import matcode
from datetime import datetime

#기본 인자 지정
check_size = (9, 6)
frame_size = (3840, 2160)
scale_factor = 10

#경로 지정
input_dir = 'input'
output_dir = 'output'

#없으면 만들기
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#각 파일의 path 설정 
left_calibration = os.path.join(input_dir, 'left_calibration.mp4')
right_calibration = os.path.join(input_dir, 'right_calibration.mp4')
left_side = os.path.join(input_dir, 'left.mp4')
right_side = os.path.join(input_dir, 'right.mp4')

def print_file_creation_date(file_path):
    creation_time = os.path.getctime(file_path)
    creation_date = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')
    print(f'{file_path} 파일이 존재합니다. 생성일: {creation_date}')

#파일이 존재하지 않을 경우, 싱글 캘리브레이션 과정 진행
if not os.path.exists('./output/left_calib_data.pkl'):
    left_calib, left_fps = calibrate_camera(left_calibration, check_size, frame_size)
    print("left video fps is :", left_fps)
    with open('./output/left_calib_data.pkl', 'wb') as f:
        pickle.dump(left_calib, f)
    print(f'왼쪽 영상 데이타 : \n{left_calib}\n저장완료')
else:
    print_file_creation_date('./output/left_calib_data.pkl')

if not os.path.exists('./output/right_calib_data.pkl'):
    right_calib, right_fps = calibrate_camera(right_calibration, check_size, frame_size)
    print("rightt video fps is :", right_fps)
    with open('./output/right_calib_data.pkl', 'wb') as f:
        pickle.dump(right_calib, f)
    print(f'오른쪽 영상 데이타 : \n{right_calib}\n저장완료')
else:
    print_file_creation_date('./output/right_calib_data.pkl')

#파일이 존재하지 않을 경우, 스테레오 캘리브레이션 과정 진행
if not os.path.exists('./output/stereo_calib_data.pkl'):
    stereo_calib = stereo_calibrate(left_calibration, right_calibration, check_size, frame_size)
    with open('./output/stereo_calib_data.pkl', 'wb') as f:
        pickle.dump(stereo_calib, f)
    print(f'스테레오 영상 데이타 : \n{stereo_calib}\n저장완료')
else:
    print_file_creation_date('./output/stereo_calib_data.pkl')

#3d 포인트 저장 및 3d 영상 저장
if not os.path.exists('./output/3d_points.pkl'):
    get_yellow_point.run(scale_factor)
    print('3d 데이타 : 출력 완료')
else:
    print_file_creation_date('./output/3d_points.pkl')
#플롯 출력
if os.path.exists('./output/3d_points.pkl'):
    matcode.run_with_matlab_engine('./output/3d_points.pkl')
    print('3d 플롯 : 출력 완료')
else:
    print('치명적 오류')