import cv2
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation

def calculate_chessboard_square_size(corners, chessboard_size):
    total_distance = 0
    num_distances = 0
    for i in range(chessboard_size[1]):
        for j in range(chessboard_size[0] - 1):
            p1 = corners[i * chessboard_size[0] + j][0]
            p2 = corners[i * chessboard_size[0] + j + 1][0]
            total_distance += np.linalg.norm(p1 - p2)
            num_distances += 1
    return total_distance / num_distances

def load_calibration_data(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"파일 '{filename}'이 존재하지 않습니다.")
        return None

def ignore_area(frame, top_fraction=0, left_fraction=0.3, bottom_fraction=0.3, right_fraction=0.3):
    height, width = frame.shape[:2]
    mask = np.ones((height, width), dtype=np.uint8) * 255
    top = int(height * top_fraction)
    bottom = int(height * (1 - bottom_fraction))
    left = int(width * left_fraction)
    right = int(width * (1 - right_fraction))
    mask[:top, :] = 0
    mask[:, :left] = 0
    mask[bottom:, right:] = 0
    return mask

def find_yellow_points(image):
    lower_yellow = np.array([20, 150, 160])
    upper_yellow = np.array([105, 255, 255])
    binary = cv2.inRange(image, lower_yellow, upper_yellow)
    points = cv2.findNonZero(binary)
    if points is not None:
        points = np.squeeze(points, axis=1)
    else:
        points = np.array([])
    return points

def extract_yellow_points(frame, mask):
    points = find_yellow_points(frame)
    centroids = []
    for (x, y) in points:
        if mask[y, x] != 0:
            centroids.append((x, y))
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
    used = set()
    for i in unique_indices:
        if i in used:
            continue
        cluster = [centroids[i]]
        for j in unique_indices:
            if i != j and np.linalg.norm(centroids[i] - centroids[j]) < threshold:
                cluster.append(centroids[j])
                used.add(j)
        merged_centroids.append(tuple(map(int, np.mean(cluster, axis=0))))
    return merged_centroids

def process_and_save_video(input_video_path, output_video_path, threshold, pickle_path):
    cap = cv2.VideoCapture(input_video_path)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    centroids_all_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        mask = ignore_area(frame)
        points = extract_yellow_points(frame, mask)
        centroids = merge_close_centroids(points, threshold)
        centroids.sort(key=lambda p: p[0])  # 좌측에서 우측 정렬

        # 검은 배경에 원 그리기
        black_bg = np.zeros(frame.shape, dtype="uint8")
        for pt in centroids:
            cv2.circle(black_bg, pt, 7, (255, 255, 255), -1)
        centroids_all_frames.append(centroids)
        out.write(black_bg)

    cap.release()
    out.release()

    with open(pickle_path, 'wb') as f:
        pickle.dump(centroids_all_frames, f)
    return centroids_all_frames

def triangulate_3d_points(left_centroids, right_centroids, calib_left, calib_right, R, T, scale_factor):
    proj1 = np.dot(calib_left['camera_matrix'], np.hstack((np.eye(3), np.zeros((3, 1)))))
    proj2 = np.dot(calib_right['camera_matrix'], np.hstack((R, T)))

    points_3D_frames = []
    for left_pts, right_pts in zip(left_centroids, right_centroids):
        # 두 프레임에서 검출된 점의 개수가 일치할 때만 삼각측량
        if len(left_pts) == len(right_pts):
            frame_3D = []
            for lp, rp in zip(left_pts, right_pts):
                pts1 = np.array(lp, dtype=np.float32).reshape(2, 1)
                pts2 = np.array(rp, dtype=np.float32).reshape(2, 1)
                p4D = cv2.triangulatePoints(proj1, proj2, pts1, pts2)
                p3D = (p4D[:3] / p4D[3]).T[0] * scale_factor
                frame_3D.append(p3D)
            points_3D_frames.append(frame_3D)
        else:
            points_3D_frames.append([])
    return points_3D_frames

def load_3d_points(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def visualize_and_save_3d_video(points3D, output_video_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    marker_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    # 모든 3D 포인트를 모아 축 범위 설정
    all_pts = np.array([p for frame in points3D for p in frame if len(frame) > 0])
    if all_pts.size:
        x_min, x_max = np.min(all_pts[:, 0]), np.max(all_pts[:, 0])
        y_min, y_max = np.min(all_pts[:, 1]), np.max(all_pts[:, 1])
        z_min, z_max = np.min(all_pts[:, 2]), np.max(all_pts[:, 2])
        margin = 5
        x_lim = (x_min - margin, x_max + margin)
        y_lim = (y_min - margin, y_max + margin)
        z_lim = (z_min - margin, z_max + margin)
    else:
        x_lim = y_lim = z_lim = (0, 1)

    def update(frame_idx):
        ax.cla()
        ax.set_title(f"Frame {frame_idx}")
        pts = points3D[frame_idx]
        for idx, p in enumerate(pts):
            if p is not None:
                p_arr = np.array(p)
                # 좌표 순서를 (X, Z, Y)로 표시 (필요에 따라 변경)
                ax.scatter(p_arr[0], p_arr[2], p_arr[1],
                           c=marker_colors[idx % len(marker_colors)],
                           label=f'Marker {idx+1}')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)
        ax.set_xlabel('X axis (mm)')
        ax.set_ylabel('Z axis (mm)')
        ax.set_zlabel('Y axis (mm)')
        if pts:
            ax.legend()

    ani = FuncAnimation(fig, update, frames=len(points3D), repeat=False)
    writer = FFMpegWriter(fps=60)
    ani.save(output_video_path, writer=writer)
    plt.close(fig)

def run(scale_factor=10):
    # 체스보드 보정을 통한 스케일 팩터 산출
    chessboard_size = (9, 6)
    cap = cv2.VideoCapture('./input/left_calibration.mp4')
    ret, frame = cap.read()
    cap.release()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            square_size_px = calculate_chessboard_square_size(corners, chessboard_size)

    # 캘리브레이션 데이터 로드
    calib_left  = load_calibration_data('./output/left_calib_data.pkl')
    calib_right = load_calibration_data('./output/right_calib_data.pkl')
    stereo_calib = load_calibration_data('./output/stereo_calib_data.pkl')
    if not (calib_left and calib_right and stereo_calib):
        print("캘리브레이션 데이터를 불러오지 못했습니다.")
        return
    R = stereo_calib['R']
    T = stereo_calib['T']

    # 비디오 처리 및 노란색 포인트 추출 (좌, 우 동일 함수 사용)
    thresh = 50  # 픽셀 단위 임계값
    left_centroids = process_and_save_video('./input/left.mp4',
                                            './output/left_output.mp4',
                                            thresh,
                                            './output/centroids_left.pkl')
    right_centroids = process_and_save_video('./input/right.mp4',
                                             './output/right_output.mp4',
                                             thresh,
                                             './output/centroids_right.pkl')

    # 3D 삼각측량
    points3D = triangulate_3d_points(left_centroids, right_centroids,
                                     calib_left, calib_right, R, T, scale_factor)
    with open('./output/3d_points.pkl', 'wb') as f:
        pickle.dump(points3D, f)

    # 3D 포인트 시각화 및 동영상 저장
    points3D = load_3d_points('./output/3d_points.pkl')
    visualize_and_save_3d_video(points3D, './output/3d_output.mp4')