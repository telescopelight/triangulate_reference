import cv2
import numpy as np
import pickle

def lucas_kanade_flow_field(input_path, output_path, centroids):
    # 동영상 파일 읽기
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    
    # 프레임의 너비, 높이, 프레임 속도 가져오기
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 동영상 저장 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # 첫 프레임 읽기
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return None
    
    # 프레임을 그레이스케일로 변환
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # 점을 Numpy 배열로 변환
    points = np.array(centroids[0], dtype=np.float32).reshape(-1, 1, 2)

    frame_idx = 1
    all_flows = []

    while True:
        # 다음 프레임 읽기
        ret, curr_frame = cap.read()
        if not ret:
            break

        # 현재 프레임을 그레이스케일로 변환
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # 옵티컬 플로우 계산
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, points, None)

        if curr_pts is not None:
            # 모든 점에 대해 플로우를 계산
            valid = status.flatten() == 1
            prev_pts = points[valid].reshape(-1, 2)
            curr_pts = curr_pts[valid].reshape(-1, 2)
            flows = (curr_pts - prev_pts).astype(np.float32)

            # 화살표 비율 조정
            scale_factor = 6.0 

            # 플로우 벡터를 현재 프레임에 그리기
            for (start, flow) in zip(prev_pts, flows):
                end_point = (int(start[0] + scale_factor * flow[0]), int(start[1] + scale_factor * flow[1]))
                cv2.arrowedLine(curr_frame, tuple(start.astype(int)), end_point, (0, 255, 0), 2)  
            
            # 프레임을 비디오에 저장
            out.write(curr_frame)

            # 모든 점의 플로우를 저장
            frame_flows = list(zip(prev_pts.tolist(), flows.tolist()))
            #print(f"Frame {frame_idx}: {frame_flows}")
            all_flows.append(frame_flows)

        # 현재 프레임을 이전 프레임으로 업데이트
        prev_gray = curr_gray

        # 다음 프레임의 점들 업데이트
        if frame_idx < len(centroids):
            points = np.array(centroids[frame_idx], dtype=np.float32).reshape(-1, 1, 2)
            frame_idx += 1

    # 자원 해제
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return all_flows

def run():
    # 예시 사용법
    input_video_path_l = './output/left_output_centroids.mp4'  # 입력 동영상 경로
    output_video_path_l = './output/left_output_flows.mp4'  # 출력 동영상 경로
    with open('./output/centroids_left.pkl', 'rb') as f:
        centroids_l = pickle.load(f)
    
    flows_l = lucas_kanade_flow_field(input_video_path_l, output_video_path_l, centroids_l)
    with open('flows_left.pkl', 'wb') as f:
        pickle.dump(flows_l, f)
    
    input_video_path_r = './output/right_output_centroids.mp4'  # 입력 동영상 경로
    output_video_path_r = './output/right_output_flows.mp4'  # 출력 동영상 경로
    with open('./output/centroids_right.pkl', 'rb') as f:
        centroids_r = pickle.load(f)
    
    flows_r = lucas_kanade_flow_field(input_video_path_r, output_video_path_r, centroids_r)
    with open('./output/flows_right.pkl', 'wb') as f:
        pickle.dump(flows_r, f)
