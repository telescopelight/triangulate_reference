import pickle
import csv

def pickle_to_csv(pickle_file_path, csv_file_path):
    # 피클 파일 로드
    with open(pickle_file_path, 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    
    # CSV 파일로 저장
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # 데이터가 리스트 형태일 때 처리
        if isinstance(data, list):
            # 리스트 내부가 리스트나 튜플로 되어 있는지 확인
            if all(isinstance(row, (list, tuple)) for row in data):
                csv_writer.writerows(data)
            else:
                # 리스트가 단일 값들로 되어 있는 경우 처리
                for item in data:
                    csv_writer.writerow([item])
        elif isinstance(data, dict):
            # 사전 데이터 처리 (키-값 페어로 저장)
            csv_writer.writerow(data.keys())
            csv_writer.writerow(data.values())
        else:
            # 기타 형식의 데이터는 직접 변환 필요
            raise ValueError("Unsupported data format in pickle file")

if __name__ == "__main__":
    pickle_file_path = '3d_points.pkl'  # 피클 파일 경로
    csv_file_path = '3d_points.csv'     # CSV 파일 경로

    pickle_file_path1 = 'centroids_left.pkl'  # 피클 파일 경로
    csv_file_path1 = 'centroids_left.csv'     # CSV 파일 경로

    pickle_file_path2 = 'centroids_right.pkl'  # 피클 파일 경로
    csv_file_path2 = 'centroids_right.csv'     # CSV 파일 경로

    pickle_to_csv(pickle_file_path2, csv_file_path2)
    print(f"Data from {pickle_file_path} has been converted to {csv_file_path}")