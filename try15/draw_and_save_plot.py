import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def extract_white_pixel_x_values(video_path, csv_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)  # 흰색 픽셀 추출

        white_pixels = np.where(thresh == 255)
        if white_pixels[0].size > 0:
            min_x = np.min(white_pixels[0])
            max_x = np.max(white_pixels[0])
        else:
            min_x = None
            max_x = None

        data.append([min_x, max_x])

    cap.release()

    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Frame', 'Min_X', 'Max_X'])
        for i, (min_x, max_x) in enumerate(data):
            csvwriter.writerow([i, min_x, max_x])

    return data

def plot_white_pixel_x_values(data, plot_path):
    frames = list(range(len(data)))
    min_x_values = [d[0] for d in data if d[0] is not None]
    max_x_values = [d[1] for d in data if d[1] is not None]

    # 최소한의 거리 설정
    distance_threshold = 10

    # 극대값과 극소값 계산 (find_peaks 사용)
    min_x_peaks, _ = find_peaks(min_x_values, distance=distance_threshold)  # min_x에서 극대값
    max_x_peaks, _ = find_peaks(max_x_values, distance=distance_threshold)  # max_x에서 극대값
    min_x_troughs, _ = find_peaks(-np.array(min_x_values), distance=distance_threshold)  # min_x에서 극소값
    max_x_troughs, _ = find_peaks(-np.array(max_x_values), distance=distance_threshold)  # max_x에서 극소값

    plt.figure(figsize=(10, 5))
    plt.plot(frames[:len(min_x_values)], min_x_values, label='Min X', color='blue')
    plt.plot(frames[:len(max_x_values)], max_x_values, label='Max X', color='red')

    # 극대값과 극소값을 플롯에 텍스트로 표시
    # 일정 간격으로 선택된 값만 표시하도록 설정
    for i, peak in enumerate(min_x_peaks):
        if i % 2 == 0:  # 5개의 값 중 하나만 표시
            plt.text(peak, min_x_values[peak], f'{int(min_x_values[peak]/5-102)}', color='green', fontsize=9, ha='right', rotation=45)
    for i, trough in enumerate(min_x_troughs):
        if i % 1 == 0:
            plt.text(trough, min_x_values[trough], f'{int(min_x_values[trough]/5-102)}', color='purple', fontsize=9, ha='right', rotation=45)

    for i, peak in enumerate(max_x_peaks):
        if i % 1 == 0:
            plt.text(peak, max_x_values[peak], f'{int(max_x_values[peak]/5-102)}', color='orange', fontsize=9, ha='right', rotation=45)
    for i, trough in enumerate(max_x_troughs):
        if i % 2 == 0:
            plt.text(trough, max_x_values[trough], f'{int(max_x_values[trough]/5-102)}', color='brown', fontsize=9, ha='right', rotation=45)

    plt.xlabel('Frame')
    plt.ylabel('X Value')
    plt.title('White Pixel X Values Over Frames')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.show()

if __name__ == "__main__":
    # 입력 및 출력 경로 설정
    reversed_video_path = 'detected1.mp4'
    csv_output_path = 'white_pixel_x_values.csv'
    plot_output_path = 'white_pixel_x_values_plot.png'

    # 흰색 픽셀 x값의 최솟값과 최대값 추출 및 저장
    data = extract_white_pixel_x_values(reversed_video_path, csv_output_path)
    print(f"White pixel x values saved to {csv_output_path}")

    # 플롯 그리기
    plot_white_pixel_x_values(data, plot_output_path)
    print(f"Plot saved to {plot_output_path}")