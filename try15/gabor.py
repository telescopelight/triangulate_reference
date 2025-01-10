import cv2
import numpy as np
import os

def apply_gabor_filter(frame, ksize=21, sigma=20, theta=np.pi/2, lambd=10.0, gamma=0.5, psi=0):
    # 가버 필터 생성
    gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
    # 가버 필터 적용
    filtered_frame = cv2.filter2D(frame, cv2.CV_8UC3, gabor_kernel)
    return filtered_frame

def mask_frame(frame, top_percent, bottom_percent, right_percent):
    height, width = frame.shape[:2]
    mask = np.ones((height, width), dtype=np.uint8) * 255

    top_mask_height = int(height * top_percent)
    bottom_mask_height = int(height * bottom_percent)
    right_mask = int(width * (1-right_percent))

    mask[:top_mask_height, :] = 0
    mask[height-bottom_mask_height:, :] = 0
    mask[:, right_mask:] = 0


    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    return masked_frame, mask

def process_video(input_path, output_path, threshold_value=30, top_percent=0.1, bottom_percent=0.1, right_percent = 0.1):
    # 입력 동영상 캡처
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {input_path}")
        return

    # 동영상 정보 가져오기
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 출력 동영상 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), isColor=True)

    # 첫 번째 프레임 읽기
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Unable to read the first frame.")
        cap.release()
        out.release()
        return

    # 그레이스케일로 변환
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        # 다음 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            break

        # 그레이스케일로 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 현재 프레임과 이전 프레임의 차이 계산
        frame_diff = cv2.absdiff(gray, prev_gray)

        # 임계값 이상인 픽셀만 남기기
        _, frame_diff_thresh = cv2.threshold(frame_diff, threshold_value, 255, cv2.THRESH_BINARY)

        # 가버 필터 적용
        gabor_filtered = apply_gabor_filter(frame_diff_thresh)

        # 마스킹 적용
        masked_frame, mask = mask_frame(gabor_filtered, top_percent, bottom_percent, right_percent)

        # 마스크된 영역을 색상으로 표시
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_color[mask == 0] = [0, 0, 0]  # 빨간색으로 마스킹된 영역 표시

        # 마스크된 영역을 원본 프레임에 합성
        combined_frame = cv2.addWeighted(cv2.cvtColor(masked_frame, cv2.COLOR_GRAY2BGR), 1, mask_color, 0.5, 0)

        # 차이 이미지를 출력 동영상에 기록
        out.write(combined_frame)

        # 현재 프레임을 이전 프레임으로 설정
        prev_gray = gray

    # 자원 해제
    cap.release()
    out.release()

def reverse_video_segment(input_path, output_path, start_sec, end_sec):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    frames = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for _ in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    reversed_frames = frames[::-1]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), isColor=True)

    for frame in reversed_frames:
        out.write(frame)

    out.release()

if __name__ == "__main__":
    # 입력 및 출력 경로 설정
    input_video_path = 'detected.mp4'
    processed_video_path = 'detected1.mp4'

    # 출력 디렉토리 없으면 생성
    if not os.path.exists('output'):
        os.makedirs('output')

    # 동영상 처리
    process_video(input_video_path, processed_video_path, threshold_value=30, top_percent=0.2, bottom_percent=0.2, right_percent = 0.4)
    print(f"Processed video saved to {processed_video_path}")
