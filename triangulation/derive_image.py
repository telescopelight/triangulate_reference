import cv2
import numpy as np
import os

def process_video(input_path, output_path, threshold_value=30):
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
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), isColor=False)

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

        # 차이 이미지를 출력 동영상에 기록
        out.write(frame_diff_thresh)

        # 현재 프레임을 이전 프레임으로 설정
        prev_gray = gray

    # 자원 해제
    cap.release()
    out.release()

if __name__ == "__main__":
    # 입력 및 출력 경로 설정
    input_video_path = 'standard.mp4'
    output_video_path = 'detected.mp4'

    # 출력 디렉토리 없으면 생성
    if not os.path.exists('output'):
        os.makedirs('output')

    # 동영상 처리
    process_video(input_video_path, output_video_path)
    print(f"Processed video saved to {output_video_path}")