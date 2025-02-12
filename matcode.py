import pickle
import numpy as np
import pandas as pd
import matlab.engine

def load_3d_points(filename='3d_points.pkl'):
    """
    피클 파일에 저장된 3D 포인트 데이터를 불러옵니다.
    """
    with open(filename, 'rb') as f:
        points3D = pickle.load(f)
    return points3D

def process_points_and_save_to_excel(points3D, fps, file_names):
    """
    각 마커에 대해 프레임 단위로 처리합니다.
      1. 유효한 포인트(여러 값이면 평균)의 좌표를 산출합니다.
      2. 전역 평균(각 축별, NaN 무시)을 계산한 후, 각 프레임의 편차(좌표-평균)를 구합니다.
      3. 편차 벡터의 L2 노름을 계산하고, z축 편차 부호를 곱해 signed displacement를 구합니다.
      4. 원본 displacement에 결측치(NaN)가 있을 경우 선형 보간하여 그 값을 대체합니다.
      5. Excel 파일의 1열에는 시간, 2열에는 보간된 displacement(숫자형),
          3열에는 원본 값이 NaN인 경우 보간된 값 오른쪽에 '#'를 덧붙인 문자열을 저장합니다.

    Excel 파일은 header와 index 없이 1행부터 바로 값이 기록됩니다.
    """
    num_frames = len(points3D)
    num_markers = max(len(frame) for frame in points3D)
    processed_files = {}

    for marker_idx, excel_file in enumerate(file_names):
        if marker_idx >= num_markers:
            print(f"Marker {marker_idx+1}는 데이터에 존재하지 않습니다.")
            continue

        marker_measurements = []
        for frame_points in points3D:
            if marker_idx < len(frame_points):
                marker_points = frame_points[marker_idx]
                valid_points = np.array([p for p in marker_points
                                         if not np.any(np.isnan(p)) and not np.any(np.isinf(p))])
                if valid_points.size > 0:
                    point = valid_points if valid_points.ndim == 1 else np.mean(valid_points, axis=0)
                    marker_measurements.append(point)
                else:
                    marker_measurements.append(np.array([np.nan, np.nan, np.nan]))
            else:
                marker_measurements.append(np.array([np.nan, np.nan, np.nan]))

        marker_measurements = np.vstack(marker_measurements)
        global_mean = np.nanmean(marker_measurements, axis=0)
        deviations = marker_measurements - global_mean
        l2_norm = np.sqrt(np.nansum(deviations**2, axis=1))
        signed_l2 = np.sign(deviations[:, 2]) * l2_norm
        time_values = np.arange(num_frames) / fps

        original_series = pd.Series(signed_l2)
        interp_series = original_series.interpolate(method='linear', limit_direction='both')
        marked_series = interp_series.astype(str)
        # 원본이 NaN인 경우, 보간된 값 뒤에 '#' 표시
        mask = original_series.isna()
        marked_series.loc[mask] = marked_series.loc[mask] + "#"

        df = pd.DataFrame({
            'Time (s)': time_values,
            'Processed Displacement (mm)': interp_series,
            'Marked Displacement (mm)': marked_series
        })
        # header와 index 없이 저장하여 Excel의 1행부터 값이 기록되게 함
        df.to_excel(excel_file, index=False, header=False)
        processed_files[excel_file] = df
        print(f"Marker {marker_idx+1}의 처리 데이터를 {excel_file}에 저장했습니다.")

    return processed_files

def run_with_matlab_engine(input_path):
    """
    3D 포인트 데이터를 불러와서 처리한 후, 두 개의 Excel 파일(예, Triangulate.xlsx와 LDV.xlsx)로 저장합니다.
    이후 MATLAB 엔진을 호출하여 통합 함수 plotAllResults를 실행합니다.
    이 함수는 FFT 분석과 센서 데이터 교차상관 분석을 한 화면(하나의 Figure)에서 나타냅니다.
    """
    points3D = load_3d_points(input_path)
    fps = 60  # 초당 프레임 수
    file_names = ['Triangulate.xlsx', 'LDV.xlsx']

    process_points_and_save_to_excel(points3D, fps, file_names)

    try:
        eng = matlab.engine.start_matlab()
        eng.addpath(r'C:\Users\telsa\Documents\MATLAB', nargout=0)
        print("MATLAB의 통합 함수 plotAllResults 호출 중...")
        eng.plotAllResults(nargout=0)
        print("MATLAB 함수 실행 완료.")
    except Exception as e:
        print(f"MATLAB 엔진 호출 중 오류 발생: {e}")