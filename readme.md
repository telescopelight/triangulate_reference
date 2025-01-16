Preparation:
	•	Each video should contain a yellow marker within the BGR range of [20, 150, 160] to [85, 255, 255].

Usage Instructions:
	1.	Input Folder Setup:
	•	A total of four .mp4 files are required for calibration and triangulation.
	•	Name the files as follows:
	•	cut_cali_left.mp4
	•	cut_cali_right.mp4
	•	left.mp4
	•	right.mp4
 
	2.	Camera Calibration:
	•	The files cut_cali_left.mp4 and cut_cali_right.mp4 should contain time-synced checkerboard frames.
	•	These checkerboard frames are used to calculate the camera matrix and stereo matrix.
 
	3.	Triangulation:
	•	The files left.mp4 and right.mp4 are used for the actual triangulation process.
 
	4.	Troubleshooting:
	•	If some 3D points are missing after the calculation, verify that all markers have been correctly detected.
	•	To address detection issues, adjust the BGR values in the get_yellow_point.py script so that the yellow marker is properly recognized.
