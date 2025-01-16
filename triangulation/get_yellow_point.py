import cv2
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
#git 확인하는 부분용으로 작성했습니다.
def calculate_chessboard_square_size(corners, chessboard_size):
    total_distance = 0
    num_distances = 0
    
    for i in range(chessboard_size[1]):
        for j in range(chessboard_size[0] - 1):
            point1 = corners[i * chessboard_size[0] + j][0]
            point2 = corners[i * chessboard_size[0] + j + 1][0]
            
            distance = np.linalg.norm(point1 - point2)
            total_distance += distance
            num_distances += 1
    
    avg_distance = total_distance / num_distances
    return avg_distance

def load_calibration_data(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"디버그: '{filename}' 파일이 존재하지 않습니다.")
        return None

def ignore_area(frame, top_fraction=0.0, left_fraction=0.4, bottom_fraction = 0.0, right_fraction = 0.25 ):
    height, width = frame.shape[:2]
    mask = np.ones((height, width), dtype=np.uint8) * 255
    top = int(height * top_fraction)
    bottom = int (height * (1 - bottom_fraction))
    left = int(width * left_fraction)
    right = int(width * (1-right_fraction))
    
    mask[:top, :] = 0
    mask[:, :left] = 0
    mask[bottom:,right:] = 0
    
    return mask

def preprocess_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def find_yellow_points(image):
    hsv = image #preprocess_image(image)
    lower_yellow = np.array([20, 150, 160])
    upper_yellow = np.array([85, 255, 255])
    
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    points = cv2.findNonZero(mask)
    if points is not None:
        points = np.squeeze(points, axis=1)
    else:
        points = np.array([])

    return points

def extract_yellow_points(frame, mask):
    points = find_yellow_points(frame)
    centroids = []

    for point in points:
        if mask[point[1], point[0]] != 0:
            centroids.append((point[0], point[1]))

    return centroids

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
        merged_centroids.sort(key=lambda x: x[0])  # 왼쪽에서 오른쪽으로 정렬
        
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
def process_and_save_video_r(input_video_path, output_video_path, threshold):
    cap = cv2.VideoCapture(input_video_path)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 동영상 저장 설정
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
        merged_centroids.sort(key=lambda x: x[0])  # 왼쪽에서 오른쪽으로 정렬
        
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
                frame_points_3D.append(points_4D.T[0] * scale_factor) ##scale_factor 적용하는 부분
            points_3D_all_frames.append(frame_points_3D)
        else:
            points_3D_all_frames.append([[], [], []])

    return points_3D_all_frames

def load_3d_points(filename):
    with open(filename, 'rb') as f:
        points3D = pickle.load(f)
    return points3D

def visualize_and_save_3d_video(points3D, output_video_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    marker_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    marker_labels = [f'Marker {i+1}' for i in range(len(marker_colors))]

    # 초기 축 범위 계산을 위한 변수
    all_points = [point for frame in points3D for marker in frame if len(marker) > 0 for point in marker]
    if len(all_points) > 0:
        all_points = np.array(all_points)
        if all_points.ndim == 1 or all_points.shape[1] == 1:
            all_points = all_points.reshape(-1, 3)  # 3차원으로 변환
        x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
        y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
        z_min, z_max = np.min(all_points[:, 2]), np.max(all_points[:, 2])
        
        # 축 범위에 여유를 두어 확대
        x_min -= 5
        x_max += 5
        y_min -= 5
        y_max += 5
        z_min -= 5
        z_max += 5

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
    writer = FFMpegWriter(fps=60)
    ani.save(output_video_path, writer=writer)
    plt.close(fig)
    
def run():
    global left_centroids, right_centroids, points_3D_all_frames
    # 체스보드의 크기 및 스케일 팩터 계산
    chessboard_size = (9, 6)
    cap = cv2.VideoCapture('./input/left_calibration.mp4')
    ret, frame = cap.read()
    cap.release()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            square_size_px = calculate_chessboard_square_size(corners2, chessboard_size)
            print(f'Average chessboard square size: {square_size_px} pixels')

            real_square_size = 25  # 실제 체스보드 사각형 한 변의 길이 (mm)
            scale_factor = real_square_size / 1
            print(f'Scale factor: {scale_factor}')
        else:
            print('Failed to find chessboard corners in the frame.')
            scale_factor = 25  # 기본 스케일 팩터 (예외 처리)
    else:
        print('Failed to read the video frame.')
        scale_factor = 25
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