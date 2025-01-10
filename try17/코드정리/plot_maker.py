import pickle
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA

def load_3d_points(filename='3d_points.pkl'):
    with open(filename, 'rb') as f:
        points3D = pickle.load(f)
    return points3D

def interpolate_nan_values(values):
    nans = np.isnan(values)
    if not np.any(~nans):
        return values  # 모든 값이 존재하지 않을 경우 생략
    indices = np.arange(len(values))
    return np.interp(indices, indices[~nans], values[~nans])

def plot_x_values_over_time(points3D, fps, peak_threshold=0.1):
    num_markers = max(len(frame) for frame in points3D)
    time_values = np.arange(len(points3D)) / fps

    plt.figure(figsize=(15, 6))
    for marker_idx in range(num_markers):
        x_values = []
        for frame_points in points3D:
            if marker_idx < len(frame_points):
                marker_points = frame_points[marker_idx]
                valid_points = np.array([p for p in marker_points if not np.any(np.isnan(p)) and not np.any(np.isinf(p))])
                if valid_points.size > 0:
                    if valid_points.ndim == 1:
                        x_value = valid_points[0]
                    else:
                        x_value = np.mean(valid_points[:, 0])
                    x_values.append(x_value)
                else:
                    x_values.append(np.nan)
            else:
                x_values.append(np.nan)
        interpolated_x_values = interpolate_nan_values(np.array(x_values))
        plt.plot(time_values, interpolated_x_values, label=f'Marker {marker_idx + 1}', color=f'C{marker_idx}')

        # 극대값과 극소값을 찾아 플롯에 표시
        peaks = (np.diff(np.sign(np.diff(interpolated_x_values))) < 0).nonzero()[0] + 1
        troughs = (np.diff(np.sign(np.diff(interpolated_x_values))) > 0).nonzero()[0] + 1
        
        # 임계값에 따른 극대값과 극소값 필터링
        for peak in peaks:
            if (interpolated_x_values[peak] - np.min(interpolated_x_values[peak-1:peak+2])) > peak_threshold:
                plt.text(time_values[peak], interpolated_x_values[peak], f'{interpolated_x_values[peak]:.2f}', color='green', fontsize=9, ha='right')
        for trough in troughs:
            if (np.max(interpolated_x_values[trough-1:trough+2]) - interpolated_x_values[trough]) > peak_threshold:
                plt.text(time_values[trough], interpolated_x_values[trough], f'{interpolated_x_values[trough]:.2f}', color='red', fontsize=9, ha='right')

    plt.xlabel('Time (s)')
    plt.ylabel('X Value (mm)')
    plt.title('Change in X Value Over Time')
    plt.grid(True)
    plt.legend()
    plt.savefig('./output/Change_in_X_Value_Over_Time.png')

def plot_y_values_over_time(points3D, fps, peak_threshold=0.1):
    num_markers = max(len(frame) for frame in points3D)
    time_values = np.arange(len(points3D)) / fps

    plt.figure(figsize=(15, 6))
    for marker_idx in range(num_markers):
        y_values = []
        for frame_points in points3D:
            if marker_idx < len(frame_points):
                marker_points = frame_points[marker_idx]
                valid_points = np.array([p for p in marker_points if not np.any(np.isnan(p)) and not np.any(np.isinf(p))])
                if valid_points.size > 0:
                    if valid_points.ndim == 1:
                        y_value = valid_points[1]
                    else:
                        y_value = np.mean(valid_points[:, 1])
                    y_values.append(y_value)
                else:
                    y_values.append(np.nan)
            else:
                y_values.append(np.nan)
        interpolated_y_values = interpolate_nan_values(np.array(y_values))
        plt.plot(time_values, interpolated_y_values, label=f'Marker {marker_idx + 1}', color=f'C{marker_idx}')

        # 극대값과 극소값을 찾아 플롯에 표시
        peaks = (np.diff(np.sign(np.diff(interpolated_y_values))) < 0).nonzero()[0] + 1
        troughs = (np.diff(np.sign(np.diff(interpolated_y_values))) > 0).nonzero()[0] + 1
        
        # 임계값에 따른 극대값과 극소값 필터링
        for peak in peaks:
            if (interpolated_y_values[peak] - np.min(interpolated_y_values[peak-1:peak+2])) > peak_threshold:
                plt.text(time_values[peak], interpolated_y_values[peak], f'{interpolated_y_values[peak]:.2f}', color='green', fontsize=9, ha='right')
        for trough in troughs:
            if (np.max(interpolated_y_values[trough-1:trough+2]) - interpolated_y_values[trough]) > peak_threshold:
                plt.text(time_values[trough], interpolated_y_values[trough], f'{interpolated_y_values[trough]:.2f}', color='red', fontsize=9, ha='right')

    plt.xlabel('Time (s)')
    plt.ylabel('Y Value (mm)')
    plt.title('Change in Y Value Over Time')
    plt.grid(True)
    plt.legend()
    plt.savefig('./output/Change_in_Y_Value_Over_Time.png')

def plot_z_values_over_time(points3D, fps, peak_threshold=0.1):
    num_markers = max(len(frame) for frame in points3D)
    time_values = np.arange(len(points3D)) / fps

    plt.figure(figsize=(15, 6))
    for marker_idx in range(num_markers):
        z_values = []
        for frame_points in points3D:
            if marker_idx < len(frame_points):
                marker_points = frame_points[marker_idx]
                valid_points = np.array([p for p in marker_points if not np.any(np.isnan(p)) and not np.any(np.isinf(p))])
                if valid_points.size > 0:
                    if valid_points.ndim == 1:
                        z_value = valid_points[2]
                    else:
                        z_value = np.mean(valid_points[:, 2])
                    z_values.append(z_value)
                else:
                    z_values.append(np.nan)
            else:
                z_values.append(np.nan)
        interpolated_z_values = interpolate_nan_values(np.array(z_values))
        plt.plot(time_values, interpolated_z_values, label=f'Marker {marker_idx + 1}', color=f'C{marker_idx}')

        # 극대값과 극소값을 찾아 플롯에 표시
        peaks = (np.diff(np.sign(np.diff(interpolated_z_values))) < 0).nonzero()[0] + 1
        troughs = (np.diff(np.sign(np.diff(interpolated_z_values))) > 0).nonzero()[0] + 1
        
        # 임계값에 따른 극대값과 극소값 필터링
        for peak in peaks:
            if (interpolated_z_values[peak] - np.min(interpolated_z_values[peak-1:peak+2])) > peak_threshold:
                plt.text(time_values[peak], interpolated_z_values[peak], f'{interpolated_z_values[peak]:.2f}', color='green', fontsize=9, ha='right')
#        for trough in troughs:
#            if (np.max(interpolated_z_values[trough-1:trough+2]) - interpolated_z_values[trough]) > peak_threshold:
#                plt.text(time_values[trough], interpolated_z_values[trough], f'{interpolated_z_values[trough]:.2f}', color='red', fontsize=9, ha='right')

    plt.xlabel('Time (s)')
    plt.ylabel('Z Value (mm)')
    plt.title('Change in Z Value Over Time')
    plt.grid(True)
    plt.legend()
    plt.savefig('./output/Change_in_Z_Value_Over_Time.png')


def calculate_initial_origin(points3D, marker_idx, num_initial_points=5):
    z_values = []
    
    for frame_points in points3D[:num_initial_points]:
        if marker_idx < len(frame_points):
            marker_points = frame_points[marker_idx]
            valid_points = np.array([p for p in marker_points if not np.any(np.isnan(p)) and not np.any(np.isinf(p))])

            if valid_points.size > 0:
                z_value = valid_points[0][2] if valid_points.ndim > 1 else valid_points[2]
                z_values.append(z_value)

    if len(z_values) == 0:
        return None  # 유효한 점이 없을 경우

    return np.mean(z_values)

def plot_marker_z_value(points3D, fps, marker_idx=4, peak_threshold=0.1, num_initial_points=5):  # marker_idx=4는 5번째 마커를 의미
    time_values = np.arange(len(points3D)) / fps
    
    # 초기 5개 프레임에서 평균 원점 계산
    origin_z = calculate_initial_origin(points3D, marker_idx, num_initial_points)

    plt.figure(figsize=(15, 6))
    
    z_values = []

    for frame_points in points3D:
        if marker_idx < len(frame_points):
            marker_points = frame_points[marker_idx]
            valid_points = np.array([p for p in marker_points if not np.any(np.isnan(p)) and not np.any(np.isinf(p))])

            if valid_points.size > 0:
                z_value = valid_points[0][2] - origin_z if valid_points.ndim > 1 else valid_points[2] - origin_z
                z_values.append(z_value)
            else:
                z_values.append(np.nan)
        else:
            z_values.append(np.nan)

    interpolated_z_values = interpolate_nan_values(np.array(z_values))
    plt.plot(time_values, interpolated_z_values, label=f'Marker {marker_idx + 1}', color=f'C{marker_idx}')

    # 극대값과 극소값을 찾아 플롯에 표시
    peaks = (np.diff(np.sign(np.diff(interpolated_z_values))) < 0).nonzero()[0] + 1
    troughs = (np.diff(np.sign(np.diff(interpolated_z_values))) > 0).nonzero()[0] + 1
    
    # 임계값에 따른 극대값과 극소값 필터링 및 텍스트 위치 조정
    for peak in peaks:
        if (interpolated_z_values[peak] - np.min(interpolated_z_values[peak-1:peak+2])) > peak_threshold:
            plt.text(time_values[peak], interpolated_z_values[peak], f'{interpolated_z_values[peak]:.2f}', 
                     color='green', fontsize=9, ha='right', va='bottom')
    for trough in troughs:
        if (np.max(interpolated_z_values[trough-1:trough+2]) - interpolated_z_values[trough]) > peak_threshold:
            plt.text(time_values[trough], interpolated_z_values[trough], f'{interpolated_z_values[trough]:.2f}', 
                     color='red', fontsize=9, ha='right', va='top')

    plt.xlabel('Time (s)')
    plt.ylabel('Z Value (mm)')
    plt.title(f'Change in Z Value Over Time for Marker {marker_idx + 1}')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'./output/Change_in_Z_Value_Over_Time_Marker_{marker_idx + 1}.png')
    plt.show()

# Example usage
# plot_marker_z_value(points3D, fps)

def plot_distances_over_time(distances, fps, start_frame):
    time_values = np.arange(start_frame, start_frame + len(distances[0])) / fps  # 시작 프레임부터 시간 값 조정
    max_distance = np.nanmax([np.nanmax(np.abs(marker_distances)) for marker_distances in distances])  # 최대 거리 계산

    for marker_idx, marker_distances in enumerate(distances):
        plt.figure(figsize=(18, 6))
        valid_indices = ~np.isnan(marker_distances)
        interpolated_distances = interpolate_nan_values(np.array(marker_distances))
        plt.plot(time_values, interpolated_distances, label=f'Marker {marker_idx + 1}', color=f'C{marker_idx}')

        # 극대값과 극소값을 찾아 플롯에 표시
        peaks = (np.diff(np.sign(np.diff(interpolated_distances))) < 0).nonzero()[0] + 1
        troughs = (np.diff(np.sign(np.diff(interpolated_distances))) > 0).nonzero()[0] + 1
        for peak in peaks:
            plt.text(time_values[peak], interpolated_distances[peak], f'{interpolated_distances[peak]:.2f}', color='green', fontsize=9, ha='right')
        for trough in troughs:
            plt.text(time_values[trough], interpolated_distances[trough], f'{interpolated_distances[trough]:.2f}', color='red', fontsize=9, ha='right')

        plt.xlabel('Time (s)')
        plt.ylabel('Displacement (mm)')
        plt.ylim(-max_distance, max_distance)
        plt.title(f'L2-Norm from Origin Over Time for Marker {marker_idx + 1}')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'./output/Displacement_from_Origin_Marker_{marker_idx + 1}.png')

def calculate_distances_from_origin(points3D):
    num_markers = max(len(frame) for frame in points3D)  # 최대 마커 수 계산
    distances = [[] for _ in range(num_markers)]
    origins = [None] * num_markers
    origins_set = False

    # 모든 마커의 첫 번째 유효한 3차원 점을 원점으로 설정
    start_frame = 0
    for frame_idx, frame_points in enumerate(points3D):
        if all(len(frame_points) > marker_idx for marker_idx in range(num_markers)) and \
           all(len(frame_points[marker_idx]) > 0 for marker_idx in range(num_markers)):
            all_valid = True
            for marker_idx in range(num_markers):
                marker_points = frame_points[marker_idx]
                valid_points = np.array([p for p in marker_points if not np.any(np.isnan(p)) and not np.any(np.isinf(p))])
                if valid_points.size > 0:
                    origins[marker_idx] = valid_points[0] if valid_points.ndim > 1 else valid_points
                else:
                    all_valid = False
                    break
            if all_valid:
                origins_set = True
                start_frame = frame_idx
                break

    if not origins_set:
        print("모든 마커가 다 존재하는 프레임을 찾지 못했습니다.")
        return distances, 0

    # 원점을 기준으로 거리 계산
    for frame_idx, frame_points in enumerate(points3D[start_frame:], start=start_frame):
        for marker_idx in range(num_markers):
            if marker_idx < len(frame_points):
                marker_points = frame_points[marker_idx]
                valid_points = np.array([p for p in marker_points if not np.any(np.isnan(p)) and not np.any(np.isinf(p))])

                if valid_points.size > 0 and origins[marker_idx] is not None:
#                    print(f'Frame: {frame_idx}, Marker: {marker_idx}, Valid Points Shape: {valid_points.shape}')
                    
                    current_point = valid_points[0] if valid_points.ndim > 1 else valid_points
                    if frame_idx<100:
                       print('index', frame_idx, 'current:', current_point, 'origin:', origins[marker_idx])
                    distance = LA.norm(current_point - origins[marker_idx])
                    
                    # If current_point's Z-coordinate is less than origin's Z-coordinate, distance is negative
                    if current_point[2] < origins[marker_idx][2]:
                        distance = -distance
                    distances[marker_idx].append(distance)
                else:
                    distances[marker_idx].append(np.nan)
            else:
                distances[marker_idx].append(np.nan)

    return distances, start_frame



def run(input_path):

    # 3D 포인트 데이터 불러오기
    points3D = load_3d_points(input_path)
    
    # 프레임 속도 설정 (초당 프레임 수)
    fps = 60
    
    # 시간에 따른 x, y, z 값 변화 플롯
    plot_x_values_over_time(points3D, fps)
    plot_y_values_over_time(points3D, fps)
    plot_z_values_over_time(points3D, fps)
    
    plot_marker_z_value(points3D, fps, marker_idx=4, peak_threshold=0.1)
    
    # 3D 좌표의 거리 변화를 계산 및 플롯
    distances, start_frame = calculate_distances_from_origin(points3D)
    plot_distances_over_time(distances, fps, start_frame)