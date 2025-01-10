import cv2
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
"""
1. 영상으로부터 마커의 중심좌표 추출
2. 추출된 중심좌표로 삼각측량 수행 및 좌표를 파일로 저장
3. 추출 좌표 확인을 위해 3D 애니메이션으로 재구성
"""

def load_calibration_data(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"디버그: '{filename}' 파일이 존재하지 않습니다.")
        return None

#마스킹 영역 지정 함수, 화면의 상하좌우로부터 비율만큼 검은색으로 처리
def ignore_area(frame, top_fraction=0.1, left_fraction=0.1, bottom_fraction = 0.2, right_fraction = 0.2 ):
    height, width = frame.shape[:2]
    mask = np.ones((height, width), dtype=np.uint8) * 255
    top = int(height * top_fraction)
    bottom = int (height * (1 - bottom_fraction))
    left = int(width * left_fraction)
    right = int(width * (1-right_fraction))
    
    mask[:top, :] = 0
    mask[:, :left] = 0
    mask[bottom:,:] = 0
    mask[:,right:] = 0

    return mask

def find_yellow_points(image):
    bgr = image #preprocess_image(image)

    #마커의 색상 범위를 정함
    lower_yellow = np.array([50, 150, 150])
    upper_yellow = np.array([100, 255, 255])
    
    mask = cv2.inRange(bgr, lower_yellow, upper_yellow)
    points = cv2.findNonZero(mask)
    if points is not None:
        points = np.squeeze(points, axis=1)
    else:
        points = np.array([])

    return points

#색상 범위에 맞는 포인트(픽셀) 추출
def extract_yellow_points(frame, mask):
    points = find_yellow_points(frame)
    centroids = []

    for point in points:
        if mask[point[1], point[0]] != 0:
            centroids.append((point[0], point[1]))

    return centroids

#추출된 마커의 픽셀들을 병합
def merge_close_centroids(centroids, threshold): 
    if not centroids:
        return []

    centroids = np.array(centroids)
    distances = np.linalg.norm(centroids[:, np.newaxis] - centroids, axis=2)
    close_pairs = np.where(distances < threshold)

    unique_indices = set()
    for i, j in zip(*close_pairs):
        if i != j:
            unique_indices.add(i)
            unique_indices.add(j)
    
    merged_centroids = []
    used_indices = set()
    for i in unique_indices:
        if i in used_indices:
            continue
        cluster = [centroids[i]]
        for j in unique_indices:
            if i != j and np.linalg.norm(centroids[i] - centroids[j]) < threshold:
                cluster.append(centroids[j])
                used_indices.add(j)
        center = np.mean(cluster, axis=0)
        merged_centroids.append(tuple(map(int, center)))
    
    return merged_centroids

#왼쪽 비디오에서 마커 추출
def process_and_save_video_l(input_video_path, output_video_path, threshold):

    cap = cv2.VideoCapture(input_video_path)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 동영상 저장 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
    
    frame_count_actual = 0
    centroid = []
    centroids = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        black_background = np.zeros(frame.shape, dtype="uint8")
        
        mask = ignore_area(frame)
        points = extract_yellow_points(frame, mask)
        merged_centroids = merge_close_centroids(points, threshold)
        merged_centroids.sort(key=lambda x: x[0])  # 왼쪽에서 오른쪽으로 픽셀 정렬
        
        for i in range(len(merged_centroids)):
            cv2.circle(black_background, merged_centroids[i], 7, (255, 255, 255), -1)
        
        centroid.append(merged_centroids)
        
        merged_centroids_padded = merged_centroids.copy()
        
        while len(merged_centroids_padded) < 5:
            merged_centroids_padded.append((0, 0))
        if len(merged_centroids_padded) > 5:
            merged_centroids_padded = merged_centroids_padded[:5]
        centroids.append(merged_centroids_padded)
        
        out.write(black_background)
        
    cap.release()
    out.release()
    
    with open('./output/centroids_left.pkl', 'wb') as f:
        pickle.dump(centroids, f)
    return centroid

#오른쪽 비디오에서 마커 추출
def process_and_save_video_r(input_video_path, output_video_path, threshold):
    cap = cv2.VideoCapture(input_video_path)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 동영상 저장 설정 - mp4 형식 저장
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
    
    frame_count_actual = 0
    centroid = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        black_background = np.zeros(frame.shape, dtype="uint8")
        
        mask = ignore_area(frame)
        points = extract_yellow_points(frame, mask)
        merged_centroids = merge_close_centroids(points, threshold)
        merged_centroids.sort(key=lambda x: x[0])  # 왼쪽에서 오른쪽으로 추출된 픽셀 정렬
        
        for i in range(len(merged_centroids)):
            cv2.circle(black_background, merged_centroids[i], 7, (255, 255, 255), -1)
        
        centroid.append(merged_centroids)
        
        out.write(black_background)
        frame_count_actual += 1
        
        for _ in range(frame_count - frame_count_actual):
            out.write(np.zeros(frame_size, dtype = "uint8"))
        
    cap.release()
    out.release()
    
    with open('./output/centroids_right.pkl', 'wb') as f:
        pickle.dump(centroid, f)
    return centroid

#각각의 점들을 삼각측량 기법을 이용해 3차원 점 좌표로 변환(Z좌표 추정)
def triangulate_3d_points(left_centroids, right_centroids, calib_data_left, calib_data_right, R, T, scale_factor):
    projMatrix1 = np.dot(calib_data_left['camera_matrix'], np.hstack((np.eye(3), np.zeros((3, 1)))))
    projMatrix2 = np.dot(calib_data_right['camera_matrix'], np.hstack((R, T)))

    points_3D_all_frames = []

    for i, (left_frame_centroids, right_frame_centroids) in enumerate(zip(left_centroids, right_centroids)):
        if len(left_frame_centroids) == len(right_frame_centroids):
            frame_points_3D = []
            for left_point, right_point in zip(left_frame_centroids, right_frame_centroids):
                pts1 = np.array(left_point, dtype=np.float32).reshape(2, 1)
                pts2 = np.array(right_point, dtype=np.float32).reshape(2, 1)
                
                points_4D_hom = cv2.triangulatePoints(projMatrix1, projMatrix2, pts1, pts2)
                points_4D = points_4D_hom[:3] / points_4D_hom[3]
                frame_points_3D.append(points_4D.T[0] * 25)
            points_3D_all_frames.append(frame_points_3D)
        else:
            points_3D_all_frames.append([[], [], []])

    return points_3D_all_frames

def load_3d_points(filename):
    with open(filename, 'rb') as f:
        points3D = pickle.load(f)
    return points3D

#저장된 3차원 마커 좌표를 이용해 애니메이션으로 변환
def visualize_and_save_3d_video(points3D, output_video_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #마커의 인덱스 0번부터 색 지정 - matplotlib에서 자동설정되는 색 순서로 설정
    marker_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    marker_labels = [f'Marker {i+1}' for i in range(len(marker_colors))]

    # 초기 축 범위 계산을 위한 변수
    all_points = [point for frame in points3D for marker in frame if len(marker) > 0 for point in marker]

    #3차원 점이 추출된 경우
    if len(all_points) > 0:
        x_lim = (-250, 250)  # (x_min, x_max)
        z_lim = (-250, 250)  # (y_min, y_max)
        y_lim = (350, 850)  # (z_min, z_max)
    else:
        x_lim = y_lim = z_lim = (0, 1)

    def update_graph(num):
        ax.cla()
        ax.set_title(f"Frame {num}")
        any_data = False
        all_points = []
        for marker_idx, points in enumerate(points3D[num]):
            if len(points) > 0:  # 마커가 비어 있지 않은지 확인
                any_data = True
                points = np.array(points)
                if points.ndim == 1:
                    points = points[np.newaxis, :]  # points가 1차원 배열이면 2차원 배열로 변환
                ax.scatter(points[:, 0], points[:, 2], points[:, 1], c=f'C{marker_idx % 10}', label=f'Marker {marker_idx + 1}')
                all_points.append(points)
                all_points.append(points)
        
        # 모든 포인트를 정렬하고 선으로 연결
        if len(all_points) > 0:
            all_points = np.vstack(all_points)
            sorted_points = all_points[np.argsort(all_points[:, 0])]
            ax.plot(sorted_points[:, 0], sorted_points[:, 2], sorted_points[:, 1], c='k')  # 검은색 선으로 연결

        if any_data:
            ax.legend()
        
        # 고정된 축 범위 설정
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)
        
        # 각 축에 레이블 추가
        ax.set_xlabel('X axis(mm)')
        ax.set_ylabel('Z axis(mm)')
        ax.set_zlabel('Y axis(mm)')

    ani = FuncAnimation(fig, update_graph, frames=len(points3D), repeat=False)
    writer = FFMpegWriter(fps=30)
    ani.save(output_video_path, writer=writer)
    plt.close(fig)
    
def run():
    global left_centroids, right_centroids, points_3D_all_frames

    # 캘리브레이션 데이터 로드
    left_calib = load_calibration_data('./output/left_calib_data.pkl')
    right_calib = load_calibration_data('./output/right_calib_data.pkl')
    stereo_calib = load_calibration_data('./output/stereo_calib_data.pkl')

    if not left_calib or not right_calib or not stereo_calib:
        print("캘리브레이션 데이터를 로드할 수 없습니다.")
        return

    R = stereo_calib['R']
    T = stereo_calib['T']

    # 비디오 처리 및 노란색 점 추출
    threshold = 50  # 픽셀 단위 임계값

    print(f"디버그: 좌측 비디오 처리 및 노란색 점 추출 시작.")
    left_centroids = process_and_save_video_l('./input/left.mp4', './output/left_output_centroids.mp4', threshold)
    print(f"디버그: 좌측 비디오의 센트로이드: {left_centroids}")

    print(f"디버그: 우측 비디오 처리 및 노란색 점 추출 시작.")
    right_centroids = process_and_save_video_r('./input/right.mp4', './output/right_output_centroids.mp4', threshold)
    print(f"디버그: 우측 비디오의 센트로이드: {right_centroids}")

    if left_centroids and right_centroids:
        points_3D_all_frames = triangulate_3d_points(left_centroids, right_centroids, left_calib, right_calib, R, T, scale_factor)
        
        if points_3D_all_frames:
            print("삼각측량 결과 3D 점들:")
            for frame_idx, points_3D in enumerate(points_3D_all_frames):
                if any(len(marker) > 0 for marker in points_3D):
                    print(f"Frame {frame_idx}: {points_3D}")
                else:
                    print(f"Frame {frame_idx}: No valid 3D points")

            with open('./output/3d_points.pkl', 'wb') as f:
                pickle.dump(points_3D_all_frames, f)
        else:
            print("디버그: 3D 점 삼각측량이 실패했습니다.")
    else:
        print("디버그: 3D 점 삼각측량을 위한 데이터가 충분하지 않습니다.")

    # 3D 포인트 불러오기
    points3D = load_3d_points('./output/3d_points.pkl')
    print(f"디버그: 로드된 3D 포인트: {points3D}")

    # 3D 포인트를 사용하여 3D 맵 동영상 저장
    visualize_and_save_3d_video(points3D, './output/3d_output.mp4')