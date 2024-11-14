#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 19:50:58 2024

@author: ramakanth
"""

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel, QLineEdit, QCheckBox
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from modules.xfeat import XFeat
import Utils as U
import torch
from segment_anything import SamPredictor, sam_model_registry




class CameraWidget(QWidget):
    def __init__(self):
        super().__init__()
        
        
        # Initialize video capture
        self.capture = self.initScope(0) #cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.video_running = False
        
        # Timer for resetting the marker color
        self.reset_color_timer = QTimer(self)
        self.reset_color_timer.setSingleShot(True)
        self.reset_color_timer.timeout.connect(self.reset_marker_color) 
        
        
        # Load icons
        self.play_icon = QIcon('play_icon.png')
        self.stop_icon = QIcon('stop_icon.png')
        
        # Marker detection results
        self.current_corners = None
        self.current_ids = None
        
        # Storing the raw images for position capturing and processing
        self.image_data = []
        self.views = []
        self.probe_pose_in_views = []
        
        # ArUco marker setup
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

        # self.dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        self.parameters = cv2.aruco.DetectorParameters()
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.parameters.cornerRefinementWinSize = 5  # Size of the window for corner refinement
        self.parameters.cornerRefinementMaxIterations = 30  # Number of iterations for refinement
        self.parameters.cornerRefinementMinAccuracy = 0.01  # Minimum accuracy for corner refinement
        
        # arucodetector
        self.arucoDetector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        k2 = [1080,1920]
        k4 = [2160,3840]
        self.frame_size = k4
        
        # Load camera calibration parameters
        self.camera_matrix,self.dist_coeffs_for_projection = U.getIntrinsic(self.frame_size,saline=False)#([2160,3840])#np.load('scopeScaledIntrinsics.npy')
        # self.dist_coeffs = np.load('/Users/ramakanth/Documents/ArthrexProjects/AREAS_Project/Arthrosocpy/synergy_calibration/distortion_coeff.npy')
        # self.camera_matrix = np.load('/Users/ramakanth/Documents/ArthrexProjects/AREAS_Project/Arthrosocpy/synergy_calibration/intrinsic.npy')
        # self.camera_matrix[0,0] = self.camera_matrix[0,0]*2
        # self.camera_matrix[1,1] = self.camera_matrix[1,1]*2
        # self.camera_matrix[0,2] = self.camera_matrix[0,2]*2
        # self.camera_matrix[1,2] = self.camera_matrix[1,2]*2
        self.dist_coeffs = self.dist_coeffs_for_projection # from the ACT teams claibration
        # mac pro front camera
        # self.dist_coeffs = np.load('/Users/ramakanth/Documents/ArthrexProjects/AREAS_Project/Arthrosocpy/macPro_calibration/mac_distortion_coeff.npy')
        # self.camera_matrix = np.load('/Users/ramakanth/Documents/ArthrexProjects/AREAS_Project/Arthrosocpy/macPro_calibration/mac_intrinsic.npy')
        
        
        print(self.camera_matrix)
        print(self.dist_coeffs)

        # Storage for captured positions and contour drawing
        self.arucoTvec = []
        self.arucoRvec = []
        self.positions = []
        self.arucoID = []
        self.contour_points_img = []
        self.contour_points = []
        self.draw_contour = False
        self.capture_frequency = 10
        self.frame_counter = 0
        
        # Initial marker color is red
        self.marker_color = (0, 0, 255)  # Red in BGR
        
        # aruco configuration
        self.marker_size = 3
        self.num_markers = 3
        self.cube_edge_offset = 1.5 # cube length offset from markersize
        self.cube_size = self.marker_size+ self.cube_edge_offset
        self.aruco_marker_ids = [6,17,10] # [2,1,0] #the printed markers
        
        # scope movement modes
        
        self.movement = False
        self.rotation_z_axis = False
        
        
        # Define marker 3D object points (local coordinates)
        self.obj_points = np.array([
            [-self.marker_size / 2, self.marker_size / 2, 0],
            [ self.marker_size / 2, self.marker_size / 2, 0],
            [ self.marker_size / 2,  -self.marker_size / 2, 0],
            [-self.marker_size / 2,  -self.marker_size / 2, 0]
        ], dtype=np.float32)
        
        # probe tip
        marker_config_3mm = {6:18,17:22,10:27}
        marker_config_4mm = {6:13,17:18.9,10:23.5}
        marker_config_paper = {2:13,1:18.6,0:22.6}
        marker_config_3mm_thick = {6:4,17:8.14,10:12.5}
        marker_config_3mm_thin = {6:5.65,17:9.59,10:13.48}
        self.probe_tip_offsets = marker_config_3mm_thick
        
        
        self.probes_aruco_ids, self.probe_tip = self.probe_config()
        
        
        # OpenCV Kalman Filter Initialization for tip position smoothing
        self.kf = cv2.KalmanFilter(6, 3)  # 6 state variables (position and velocity), 3 measurement variables (x, y, z)

        # State transition matrix (F)
        self.kf.transitionMatrix = np.array([[1, 0, 0, 1, 0, 0],  # X + Vx
                                             [0, 1, 0, 0, 1, 0],  # Y + Vy
                                             [0, 0, 1, 0, 0, 1],  # Z + Vz
                                             [0, 0, 0, 1, 0, 0],  # Vx remains
                                             [0, 0, 0, 0, 1, 0],  # Vy remains
                                             [0, 0, 0, 0, 0, 1]], np.float32)

        # Measurement matrix (H)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],  # Measure X
                                              [0, 1, 0, 0, 0, 0],  # Measure Y
                                              [0, 0, 1, 0, 0, 0]], np.float32)

        # Process noise covariance (Q)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-4

        # Measurement noise covariance (R)
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-2

        # Error covariance (P)
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)

        # Initial state
        self.kf.statePost = np.zeros((6, 1), np.float32)
        
        
        
        
        # feature method
        self.feature_method = "xFeat" #"ORB","xFeat"
        if self.feature_method=="xFeat":
            self.xfeat = XFeat()
        else:
            self.orb = cv2.ORB_create()
            
        self.matcher_for_xfeat = False
            
        self.segment_method = 'BB' # 'BB', 'SAM'
        # self.xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = 4096)
       
        
        self.predictor = None
        # segmentaion model SAM
        if self.segment_method == 'SAM':
            self.load_SAM()
            
        
        self.initUI()
        
    def load_SAM(self):
        sam_checkpoint = "/Users/ramakanth/Documents/ArthrexProjects/AREAS_Project/Arthrosocpy/localCam/accelerated_features/sam_vit_b_01ec64.pth"
        model_type = "vit_b"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.predictor = SamPredictor(sam)
        
    def probe_config(self):
        
        if self.num_markers == 1:
            # Known marker IDs on the probe
            known_marker_ids = [self.aruco_marker_ids[0]]
            probe_tip_1 = np.array([[0,self.probe_tip_offsets[known_marker_ids[0]],self.cube_size/2]])
            probe_tip = { self.aruco_marker_ids[0]: probe_tip_1}
        elif self.num_markers == 2:
            # Known marker IDs on the probe
            known_marker_ids = self.aruco_marker_ids[1:3]
            probe_tip_1 = np.array([[0,self.probe_tip_offsets[known_marker_ids[0]],self.cube_size/2]])
            probe_tip_2 = np.array([[0,self.probe_tip_offsets[known_marker_ids[1]],self.cube_size/2]])
            probe_tip = {self.aruco_marker_ids[1]:probe_tip_1,self.aruco_marker_ids[2]:probe_tip_2}
        elif self.num_markers == 3:
            # Known marker IDs on the probe
            known_marker_ids = self.aruco_marker_ids
            probe_tip_1 = np.array([[0,self.probe_tip_offsets[known_marker_ids[0]],self.cube_size/2]])
            probe_tip_2 = np.array([[0,self.probe_tip_offsets[known_marker_ids[1]],self.cube_size/2]])
            probe_tip_3 = np.array([[0,self.probe_tip_offsets[known_marker_ids[2]],self.cube_size/2]])
            probe_tip = {self.aruco_marker_ids[0]:probe_tip_1,self.aruco_marker_ids[1]:probe_tip_2, self.aruco_marker_ids[2]:probe_tip_3}
        # print(probe_tip)
        return known_marker_ids, probe_tip
    
    def initUI(self):
        self.setWindowTitle('ArUco Marker Measurement System')
        self.setGeometry(100, 100, 800, 600)
        layout = QGridLayout()
        
        
       
        
        
        # Checkboxes for camera matrix and distortion coefficients
        intrinsics_selection_bar = QHBoxLayout()
        
        self.camera_matrix_checkbox = QCheckBox("Calibrated Camera Matrix", self)
        self.camera_matrix_checkbox.toggled.connect(self.load_calibrated_camera_matrix)
        intrinsics_selection_bar.addWidget(self.camera_matrix_checkbox)
        

        self.distortion_coeff_checkbox = QCheckBox("Calibrated Distortion Coefficients", self)
        self.distortion_coeff_checkbox.toggled.connect(self.load_calibrated_distortion_coefficients)
        intrinsics_selection_bar.addWidget(self.distortion_coeff_checkbox)
        
        
        layout.addLayout(intrinsics_selection_bar, 0, 0)  # Row 1, Column 0

        
        # air vs saline
        self.saline_intrinsics_checkbox = QCheckBox('Intrinsics in Saline',self)
        self.saline_intrinsics_checkbox.toggled.connect(self.load_saline_intrinsics)
        layout.addWidget(self.saline_intrinsics_checkbox,0,1)# row 1, column 3

        # mode_selection_bar
        mode_selection_bar = QHBoxLayout()
        
        self.movement_checkbox = QCheckBox("Movement_Mode", self)
        self.movement_checkbox.toggled.connect(self.movement_mode_activation)
        mode_selection_bar.addWidget(self.movement_checkbox)
        
        self.rotation_checkbox = QCheckBox("Z_axis_Rotation_Mode", self)
        self.rotation_checkbox.toggled.connect(self.rotation_mode_activation)
        mode_selection_bar.addWidget(self.rotation_checkbox)
        
        layout.addLayout(mode_selection_bar, 0, 2)
         

        # Settings button at the top-left corner
        self.settings_button = QPushButton('Settings', self)
        self.settings_button.clicked.connect(self.openSettings)
        layout.addWidget(self.settings_button,1,0,1,3)  #
        
        
        self.video_label = QLabel(self)
        layout.addWidget(self.video_label,2,0,1,3)
        
        

        self.toggle_video_button = QPushButton('', self)
        self.toggle_video_button.setIcon(self.play_icon)
        self.toggle_video_button.clicked.connect(self.toggle_video)
        layout.addWidget(self.toggle_video_button,3,0,1,3)
        
        self.capture_button = QPushButton('Capture Position', self)
        self.capture_button.clicked.connect(self.capture_position)
        layout.addWidget(self.capture_button,4,0)
        
        
        # Task bar layout for position indicators
        task_bar_layout = QHBoxLayout()
        
        self.pos1_indicator = QLabel('Position 1: Not Captured')
        self.pos1_indicator.setStyleSheet("background-color: red; color: white; padding: 5px;")
        task_bar_layout.addWidget(self.pos1_indicator)

        self.pos2_indicator = QLabel('Position 2: Not Captured')
        self.pos2_indicator.setStyleSheet("background-color: red; color: white; padding: 5px;")
        task_bar_layout.addWidget(self.pos2_indicator)
        
        layout.addLayout(task_bar_layout,4,1)

        # distance diaplay field
        self.distance_display = QLineEdit(self)
        self.distance_display.setReadOnly(True)
        layout.addWidget(self.distance_display,4,2)

        self.draw_contour_button = QPushButton('Draw Contour', self)
        self.draw_contour_button.setCheckable(True)
        self.draw_contour_button.toggled.connect(self.toggle_contour_drawing)
        layout.addWidget(self.draw_contour_button,5,0)
        
        self.area_display = QLineEdit(self)
        self.area_display.setReadOnly(True)
        layout.addWidget(self.area_display,5,1,1,2)
        
        self.setLayout(layout)
        
    def initScope(self,camid):
        # Initialize the video capture object with the device index
        cap = cv2.VideoCapture(camid)  # 0 is usually the default camera
        
        if not cap.isOpened():
            print("Error: Could not open video device.")
        else:
            # Set frame width and height
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        
            # Set frame rate
            cap.set(cv2.CAP_PROP_FPS, 30)
        return cap
        
    def openSettings(self):
        self.settings_window = U.SettingsWindow()
        self.settings_window.settings_updated.connect(self.applySettings)
        self.settings_window.show()

    def applySettings(self, settings):
        
        self.frame_size = settings['frame_size']
        
        self.camera_matrix = U.getIntrinsic(settings['frame_size'],saline=False)
        
        self.probe_tip_offsets = settings['probe_tip_offsets']
        
        self.feature_method = settings['feature_method']
        
        self.num_markers = settings['aruco_configuration'][0]
        
        self.marker_size = settings['aruco_configuration'][1]
        
        self.segment = settings['segement_method']
        
        print(f"Settings received: {settings}")
        
    def toggle_video(self):
        if not self.video_running:
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)  # Update every ~33 ms
            self.toggle_video_button.setIcon(self.stop_icon)
            # self.toggle_video_button.setText('Stop Video')
            self.video_running = True
        else:
            self.timer.stop()
            self.timer.timeout.disconnect(self.update_frame)
            self.video_label.clear()
            self.toggle_video_button.setIcon(self.play_icon)
            # self.toggle_video_button.setText('Start Video')
            self.video_running = False
            
    def keyPressEvent(self, event):
        # Foot pedal as a key press to capture position (e.g., space bar)
        if event.key() == Qt.Key_B: # Qt.Key_Space
            self.capture_position()
            print('keypressed')
    
    def load_calibrated_camera_matrix(self,checked):
        
        if checked:
            self.camera_matrix = np.load('/Users/ramakanth/Documents/ArthrexProjects/AREAS_Project/Arthrosocpy/synergy_calibration/intrinsic.npy')
            # self.camera_matrix[0,0] = self.camera_matrix[0,0]*2
            # self.camera_matrix[1,1] = self.camera_matrix[1,1]*2
            # self.camera_matrix[0,2] = self.camera_matrix[0,2]*2
            # self.camera_matrix[1,2] = self.camera_matrix[1,2]*2
            print(self.camera_matrix)
        else:
            self.camera_matrix, _ = U.getIntrinsic(self.frame_size)
            
    def load_calibrated_distortion_coefficients(self,checked):
        
        if checked:
            self.dist_coeffs = np.load('/Users/ramakanth/Documents/ArthrexProjects/AREAS_Project/Arthrosocpy/synergy_calibration/distortion_coeff.npy')
            print(self.dist_coeffs)
        else:
            _, self.dist_coeffs =  U.getIntrinsic(self.frame_size)
        
    def load_saline_intrinsics(self,checked):
        
        if checked:
            self.camera_matrix, _ = U.getIntrinsic(self.frame_size,saline=True)
            print(self.camera_matrix)
        else:
            self.camera_matrix, _ = U.getIntrinsic(self.frame_size,saline=False)
      
    def movement_mode_activation(self,checked):
        
        if checked:
            
            self.movement = True
        else:
            
            self.movement = False
            
    def rotation_mode_activation(self, checked):
        
        if checked:
            self.rotation_z_axis = True
        else:
            self.rotation_z_axis = False
    def display_frame(self, frame):
        # Convert frame to display on QLabel
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        # print("{} width, {} height".format(w,h))
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.video_label.size(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(p))
        
    def update_frame(self):
        ret, self.frame = self.capture.read()
        # print(self.frame.shape)
        if ret:
            self.store_frame(self.frame)
            # Get the latest frame data to display (after storing)
            data = self.image_data[-1]
            frame = data['frame']
            
            if self.draw_contour_button.isChecked():
                frame = self.detect_contour(frame, data['corners'], data['ids'])
              
            else:
                frame = self.detect_and_draw_markers(frame, data['corners'], data['ids'])
            self.display_frame(frame)

   

    
    def detect_and_draw_markers(self, frame,corners, ids):
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # corners, ids, rejectedImgPoints = self.arucoDetector.detectMarkers(gray)
        # # corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)
        # self.current_corners = corners
        # self.current_ids = ids
        # print(self.current_ids)
        # print(self.current_corners)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)  # Draw marker internal squares
            for corner in corners:
                pts = corner.reshape(-1, 2)
                cv2.polylines(frame, [np.int32(pts)], True, self.marker_color, 5)
        return frame
    
    def reset_marker_color(self):
        # Reset marker color to red
        self.marker_color = (0, 0, 255)  # Red
        self.update_frame()  # Optionally update the frame to show the color change immediately


    # def capture_position(self):
    #     # Processing code to be implemented
    #     pass
    
    def toggle_contour_drawing(self,checked):
        # print(checked)
        if checked:
            self.contour_points = []  # Initialize/reset the list of contour points
            self.contour_points_img = []
            self.frame_counter = 0 
            self.draw_contour_button.setText('Stop Drawing Contour')
        else:
            if len(self.contour_points) > 2:
                self.draw_contour_button.setText('Draw Contour')
                points = np.vstack(self.contour_points)
                area = self.project_points_using_pca_and_compute_area(points)
                self.area_display.setText(f"Area: {area:.2f} mm^2")
            else:
                self.area_display.setText("Area: 0.00 mm^2")


       
        
    
    
    def detect_contour(self, frame,corners,ids):
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # corners, ids, rejectedImgPoints = self.arucoDetector.detectMarkers(gray)
        rvec_probe_cc = np.zeros((3, 1))
        tvec_probe_cc =  np.zeros((3, 1))

        # corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)
        if ids is not None:
            markers_of_interest = self.detected_ids(ids)
            
            if markers_of_interest is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                # self.current_corners = corners
                # Capture probe tip point every `capture_frequency` frames
                if self.frame_counter % self.capture_frequency == 0:
                    ret ,rvec , tvec,  = self.estimate_probe_pose(corners,ids,markers_of_interest)
                    
                    # Kalman Filter Prediction
                    predicted_state = self.kf.predict()  # Predict the next position
    
                    # Update the Kalman filter with the new measurement (tip position)
                    measurement = np.array(tvec, dtype=np.float32).reshape(3, 1)
                    self.kf.correct(measurement)
    
                    # Get the filtered position from the Kalman filter state
                    filtered_position = self.kf.statePost[:3].flatten()  # Extract position (x, y, z)

                    self.contour_points.append(filtered_position)
                    
                # if self.draw_contour_button.isChecked() and self.current_corners is not None:
                    # Assuming single marker detection for simplicity
                    # center = np.mean(self.current_corners[0][0], axis=0)
                    tip, _ = cv2.projectPoints(filtered_position, rvec_probe_cc, tvec_probe_cc, self.camera_matrix, self.dist_coeffs)
    
                    # tip = self.project_to_image_plane(tip_position[0], self.camera_matrix)
                
                    self.contour_points_img.append(np.squeeze(tip))
                    
                self.frame_counter += 1
                if len(self.contour_points_img) > 1:
                    
                    smoothed_points = self.smooth_contours(self.contour_points_img)
                    
                        
                    # Draw lines between consecutive smoothed points
                    for i in range(1, len(smoothed_points)):
                        pt1 = tuple(smoothed_points[i - 1].astype(int))
                        pt2 = tuple(smoothed_points[i].astype(int))
                        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)  # Green line
        return frame
    
    def smooth_contours(self, points, window_size=3):
        """Apply a simple moving average to smooth the points"""
        if len(points) < window_size:
            return points  # Not enough points to apply smoothing
        
        smoothed_points = []
        points_array = np.array(points)
        for i in range(len(points) - window_size + 1):
            window = points_array[i:i + window_size]
            smoothed_point = np.mean(window, axis=0)
            smoothed_points.append(smoothed_point)
        
        return smoothed_points
    
    def add_mask(self,image, mask):
        # Overlay the mask on the image (red color to indicate the filtered region)
        mask_overlay = np.zeros_like(image, dtype=np.uint8)
        mask_overlay[mask == 255] = [0, 0, 255]  # Red color where the mask is
        
        # Blend the original image with the mask overlay for visualization
        alpha = 0.5  # Transparency factor
        visualized_image = cv2.addWeighted(image, 1 - alpha, mask_overlay, alpha, 0)
        return visualized_image
    
    def capture_position(self):
        
        rvec_probe_cc = np.zeros((3, 1))
        tvec_probe_cc =  np.zeros((3, 1))
        
        if self.image_data:
            # Get the latest frame data
            data = self.image_data[-1]
            corners = data['corners']
            ids = data['ids']
            frame = data['frame']
        # Detect ArUco markers
        # plt.imshow(self.frame)
        # gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        # corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)
        # Reset if starting a new measurement
            if len(self.positions) == 2:
                self.positions.clear()
                self.views.clear()
                self.probe_pose_in_views.clear()
                self.distance_display.clear()
                self.pos1_indicator.setText('Position 1: Not Captured')
                self.pos1_indicator.setStyleSheet("background-color: red; color: white; padding: 5px;")
                self.pos2_indicator.setText('Position 2: Not Captured')
                self.pos2_indicator.setStyleSheet("background-color: red; color: white; padding: 5px;")
            
        
        
        
            if ids is not None:
                markers_of_interest = self.detected_ids(ids)
                print(ids)
                if markers_of_interest is not None:
                    
                    ret, rvec, tvec  = self.estimate_probe_pose(corners, ids, markers_of_interest)
                    
                    tip, _ = cv2.projectPoints(tvec.T, rvec_probe_cc, tvec_probe_cc, self.camera_matrix, self.dist_coeffs)
                    
    
                    self.positions.append(tvec.T)
                    self.probe_pose_in_views.append({'t':tvec,'R':rvec,'id':None})
                    self.views.append(frame)
                    
                    if len(self.positions) == 1:
                        # Update indicator for position 1
                        self.pos1_indicator.setText('Position 1: Captured')
                        self.pos1_indicator.setStyleSheet("background-color: green; color: white; padding: 5px;")
                        
                    elif len(self.positions) == 2:
                        # Update indicator for position 2 and calculate distance
                        self.pos2_indicator.setText('Position 2: Captured')
                        self.pos2_indicator.setStyleSheet("background-color: green; color: white; padding: 5px;")
                        R, t ,R_H, mask1, mask2 = self.process_camera_movement(self.views[-1], self.views[-2],self.probe_pose_in_views[-1],self.probe_pose_in_views[-2])
                        self.measure_distance(R, t, R_H, movement=self.movement,rotation=self.rotation_z_axis)
                        
                        # print(R)
                       
                    # Change marker color to green upon capture
                    self.marker_color = (0, 255, 0)  # Green in BGR
                    cv2.circle(frame, (int(tip[0,0,0]),int(tip[0,0,1])), 20, (255, 0, 255), -1)
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 20,thickness=10)
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    plt.imshow(frame_rgb)
                    plt.show()
                    self.reset_color_timer.start(500)  # Reset color after 500 ms
            self.display_frame(frame)
        
    def detected_ids(self,current_ids):
        
        # Filter out only the known markers
        # if self.current_ids is not None:
        detected_ids = current_ids.flatten()
        # print(detected_ids)

        markers_of_interest = [i for i in range(len(detected_ids)) if detected_ids[i] in self.probes_aruco_ids]
        # print(markers_of_interest)
        if markers_of_interest:
            return markers_of_interest
        else:
            return None
        
    def estimate_probe_pose2(self,corners,ids,markers_of_interest):
 
        rvecs = []
        tvecs = []
        tip_positions = []
        probe_ids = []

        for idx in markers_of_interest:
            # print(idx)
            marker_corners = corners[idx][0]
            # print(marker_corners)
            # print(obj_points)
            # print(self.camera_matrix)
            marker_in_probe_space = self.obj_points + self.probe_tip[ids[idx][0]]
            # Estimate the pose of the current marker
            ret, rvec, tvec = cv2.solvePnP(marker_in_probe_space, marker_corners, self.camera_matrix, self.dist_coeffs,cv2.SOLVEPNP_IPPE_SQUARE)
            if ret:
                # rvecs[ids[idx][0]] = rvec
                # tvecs[ids[idx][0]] = tvec
                rvecs.append(rvec)
                tvecs.append(tvec)
                probe_ids.append(ids[idx][0])
                
                print(tvec) # shape = (3,1)
                print(rvec)
                # projected_points, _ = cv2.projectPoints(obj_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
                # print(projected_points)
                
                # projected_points = self.project_to_image_plane(self.probe_tip[ids[idx][0]].reshape(3,1), self.camera_matrix, rvec=rvec, tvec=tvec)
                # print(projected_points)
                # pb_tp = self.project_to_camera_coord(self.probe_tip[ids[idx][0]].reshape(3,1), rvec, tvec)
                # print(pb_tp)
                # tip_positions.append(pb_tp.reshape(1,3))
                tip_positions.append(tvec.reshape(1,3))

                # print(np.linalg.norm(tvec - pb_tp))
                
        # Step 3: Improve tip detection using multiple tvecs (if more than one marker detected)
        # print(tip_positions)
        if len(tvecs) > 1:
            # Average tvecs to improve the position estimate of the probe tip
            # Since the markers are linearly arranged, taking an average helps to reduce error
            tp = np.mean(tip_positions, axis=0) # shape = (1,3)
            # print(tp.shape)
            # self.positions.append(tp)
            # self.arucoTvec.append(tvecs[0])
            # self.arucoRvec.append(rvecs[0])
            # print("Improved Probe Tip Position (in object space):", tip_position.flatten())
            # print(tp.shape)
            return tp, rvecs, tvecs, probe_ids
        else:
            
            # self.positions.append(tip_positions[0])
            # self.arucoTvec.append(tvecs[0])
            # self.arucoRvec.append(rvecs[0])

            # # If only one marker is detected, return its tvec as the estimate for the probe tip position
            # print("Single Marker Detected: Using Single Marker Pose")
            # print(tip_positions[0].shape) # shape = (1,3)
            return tip_positions[0], rvecs, tvecs, probe_ids
    

    def estimate_probe_pose(self,corners,ids,markers_of_interest):
 
        # Assume we have N markers detected in the frame
        objPoints_combined = []
        imgPoints_combined = []

        for idx in markers_of_interest:
            # print(idx)
            marker_corners = corners[idx][0]
            
            marker_in_probe_space = self.obj_points + self.probe_tip[ids[idx][0]]
            
            
            
            # Add to the combined lists
            objPoints_combined.extend(marker_in_probe_space)
            imgPoints_combined.extend(marker_corners)
            
        # Convert combined points to NumPy arrays of appropriate shape
        objPoints_combined = np.array(objPoints_combined, dtype=np.float32)
        imgPoints_combined = np.array(imgPoints_combined, dtype=np.float32)
        # print(objPoints_combined)
        # print(imgPoints_combined)

        # Estimate pose for the entire probe
        ret, rvec_probe, tvec_probe = cv2.solvePnP(objPoints_combined, imgPoints_combined, self.camera_matrix, self.dist_coeffs)
        rvec_probe,tvec_probe = cv2.solvePnPRefineLM(objPoints_combined, imgPoints_combined, self.camera_matrix, self.dist_coeffs, rvec_probe, tvec_probe)
        print(tvec_probe) 
        print(rvec_probe)
            
        return ret, rvec_probe, tvec_probe
        
    def apply_transform(self,R,t,p):
        
        # Create a full transformation matrix from R and t
        transformation_matrix = np.eye(4)  # Create a 4x4 identity matrix
        transformation_matrix[:3, :3] = R  # Set the upper left 3x3 block to be the rotation matrix R
        transformation_matrix[:3, 3] = t.ravel()  # Set the last col
        
        p_homogeneous = np.append(p, 1)
        
        # marker_position_frame2 = np.array([[x, y, z, 1]]).T  # Homogeneous coordinates
        # marker_position_frame1 = R @ marker_position_frame2[:3] + t  #
        
        # Apply the transformation
        new_position_homogeneous = np.dot(transformation_matrix, p_homogeneous)  # Matrix multiplication
    
        # Convert back from homogeneous coordinates to 3D by removing the last element
        new_position = new_position_homogeneous[:3]
        return new_position
    def project_to_camera_coord(self, points, rvec, tvec):
        print(points)
        print(tvec)
        print(rvec)
        
        if points.shape[1] == 3:
            points = points.T  # Convert to (3, N)
        
        # Step 1: Convert the rotation vector to a rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        print(rotation_matrix)
        
        # Step 2: Transform the probe tip from the marker coordinate system to the camera coordinate system
        # Apply the rigid transformation: P_camera = R * P_marker + t
        P_c = np.dot(rotation_matrix, points) + tvec
        return P_c
        
    def project_to_image_plane(self,points, camera_matrix, rvec=None , tvec=None):
        # Ensure that points are in (3, N) shape
        # print(points.shape)
        if points.shape[1] == 3:
            points = points.T  # Convert to (3, N)
        
        if rvec is not None:
           points = self.project_to_camera_coord(points, rvec, tvec)

        # Ensure P_c is in homogeneous coordinates (assuming P_c is given as (X, Y, Z))
        N = points.shape[1]
        # Add a row of 1's to convert to homogeneous coordinates
        ones_row = np.ones((1, N), dtype=np.float32)  # Row of 1's with shape (1, N)
        points_homogeneous = np.vstack((points, ones_row))  # Shape becomes (4, N)
        
        # P_c_homogeneous = np.array([P_c[0,0], P_c[1,0], P_c[2,0], 1.0])
        
        # Extend the camera matrix to a projection matrix of shape (3, 4)
        projection_matrix = np.hstack((camera_matrix, np.zeros((3, 1), dtype=np.float32)))
        
        # Multiply by the camera matrix (Note: We only use the first three coordinates for multiplication)
        image_points_homogeneous = np.dot(projection_matrix, points_homogeneous)
        
        # Convert from homogeneous image coordinates to 2D pixel coordinates
        # x_pixel = p[0] / p[2]
        # y_pixel = p[1] / p[2]
        
        x_pixel = image_points_homogeneous[0, :] / image_points_homogeneous[2, :]
        y_pixel = image_points_homogeneous[1, :] / image_points_homogeneous[2, :]
        
        return np.vstack((x_pixel, y_pixel)).T
    
    def measure_distance(self,R,t, R_H, movement=False,rotation=False):
        p1, p2 = self.positions[0], self.positions[1]
        
        if movement:
            
            p2 = self.apply_transform(R, t, p2)
            
        if rotation:
            
            p2 = self.apply_transform(R_H, np.zeros(3,1), p2)
        # p1 = p1+self.probe_tip # tip offset added
        # p2 = p2+self.probe_tip # tip offset added
        
        
        distance = np.linalg.norm(p1 - p2)
        self.distance_display.setText(f"Distance: {distance:.2f} mm")
        # self.positions = []  # Reset after displaying distance
        # self.views = [] # remove views
        # self.arucoTvec = []
        # self.arucoRvec = []
        # print(self.current_ids)
        self.marker_color = (0, 0, 255)  # Red
        
    def store_frame(self, frame):
        # # Append the current frame to the images list and keep only the last three frames
        # self.images.append(frame.copy())
        # if len(self.images) > 3:
        #     self.images.pop(0)  # Remove the oldest frame if there are more than three frames
        
        # Convert to grayscale for marker detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = self.arucoDetector.detectMarkers(gray)
        # Store the frame along with its corresponding marker data
        image_data = {
            'frame': frame.copy(),
            'corners': corners,
            'ids': ids
        }
        self.image_data.append(image_data)
    
        # Limit the stored frames to the last 3
        if len(self.image_data) > 3:
            self.image_data.pop(0)
  

    def update_contour(self):
        # Similar detection logic, add points to contour_points
        pass
    
    def calculate_contour_area(self):
        # Use PCA to project points onto a 2D plane and calculate the area
        
        pass
    
    def process_camera_movement(self, img1, img2, pose_in_img1, pose_in_img2):
        if self.feature_method =='xFeat':
            xFeat = True
        else:
            xFeat = False
            
        if self.segment_method=='SAM':
            segment = True
        else:
            segment = False
            
        
        
        if xFeat:
            
            if  self.matcher_for_xfeat:
                
                kp1, des1, mask1, img1 = self.extractFeatures(img1, pose_in_img1, xFeat=xFeat, segment=segment)
                kp2, des2, mask2, img2  = self.extractFeatures(img2, pose_in_img2, xFeat=xFeat, segment=segment)
                
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                
                # Match descriptors
                matches = bf.match(des1, des2)
            
                # Sort them in the order of their distance
                matches = sorted(matches, key=lambda x: x.distance)
            
                # Extract the matched points
                points1 = np.zeros((len(matches), 2), dtype=np.float32)
                points2 = np.zeros((len(matches), 2), dtype=np.float32)
            
                for i, match in enumerate(matches):
                    points1[i, :] = kp1[match.queryIdx]
                    points2[i, :] = kp2[match.trainIdx]
                    
                # print(len(matches))
                # print(points1.shape)
                # print(points2.shape)

            else:
                mask1, mask_3C1 = self.get_image_masks(img1, pose_in_img1,segment=segment)
                img1 = cv2.bitwise_and(mask_3C1,img1)
                
                mask2, mask_3C2 = self.get_image_masks(img2, pose_in_img2,segment=segment)
                img2 = cv2.bitwise_and(mask_3C2,img2)
                kp1, kp2 = self.xfeat.match_xfeat(img1, img2, top_k = 4096)
                combined_mask = cv2.bitwise_and(mask1, mask2)
                # points1 = self.filterKeyPoints(kp1, mask1, xFeat=xFeat)
                # points2 = self.filterKeyPoints(kp2, mask2, xFeat=xFeat)
                points1, points2 = self.filterMatchedKeyPoints(kp1, kp2, combined_mask)
                
                # _, mask3 = cv2.findHomography(kp1, kp2, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
                # mask3 = mask3.flatten()
                # matches = [cv2.DMatch(i,i,0) for i in range(len(mask3)) if mask3[i]]
                matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(points1))]
                # print(len(matches))
                # print(points1.shape)
                # print(points2.shape)

                # print(points1.dtype)

            
        else :  
            
            kp1, des1, mask1, img1 = self.extractFeatures(img1, pose_in_img1, xFeat=xFeat, segment=segment)
            kp2, des2, mask2, img2  = self.extractFeatures(img2, pose_in_img2, xFeat=xFeat, segment=segment)
            # Create BFMatcher object
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
            # Match descriptors
            matches = bf.match(des1, des2)
        
            # Sort them in the order of their distance
            matches = sorted(matches, key=lambda x: x.distance)
        
            # Extract the matched points
            points1 = np.zeros((len(matches), 2), dtype=np.float32)
            points2 = np.zeros((len(matches), 2), dtype=np.float32)
        
            for i, match in enumerate(matches):
                points1[i, :] = kp1[match.queryIdx]
                points2[i, :] = kp2[match.trainIdx]
    
            # print(len(matches))
            # print(points1.shape)
            # print(points1.dtype)
        
    
        # Prepare keypoints and matches for drawMatches function
        keypoints1 = [cv2.KeyPoint(p[0], p[1],  5) for p in kp1]
        keypoints2 = [cv2.KeyPoint(p[0], p[1],  5) for p in kp2]
        # Compute the fundamental matrix
        F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)
        
        # Decompose homography into rotation, translation, and normal (ignoring translation since camera is static)
        U, S, Vt = np.linalg.svd(F)
        R_H = np.dot(U, Vt)
        
        # Calculate the rotation angle around the z-axis
        theta = np.arctan2(R_H[1, 0], R_H[0, 0])
        theta_deg = np.degrees(theta)
        
        print(f"Rotation angle around the optical axis (z-axis) in degrees: {theta_deg:.2f}")
            
        # Compute the essential matrix
        E, mask = cv2.findEssentialMat(points1, points2, self.camera_matrix, cv2.RANSAC, prob=0.999, threshold=1.0)
    
    
        # Select inlier points
        points1_inliers = points1[mask.ravel() == 1]
        points2_inliers = points2[mask.ravel() == 1]
        
        # print(points1_inliers)
        

        # Recover pose
        _, R, t, mask = cv2.recoverPose(E, points1, points2, self.camera_matrix)
    
        # Calculate the rotation angle in radians
        theta = np.arccos((np.trace(R) - 1) / 2)
        
        # Convert the angle from radians to degrees
        theta_deg = np.degrees(theta)
        
        print(f"Rotation_camera angle (in degrees): {theta_deg:.2f}")
    
        # Here you would apply the rotation and translation to adjust the marker positions or for further processing
        # This could involve transforming 3D points or rectifying images
        # print(len(matches))
        # print(len(keypoints1))
        # print(len(keypoints2))
        
        matched_image = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(matched_image)
        plt.show()

        return R, t, R_H, mask1, mask2
        print(f"Rotation Matrix:\n{R}\nTranslation Vector:\n{t}")
    
    
    
    
    def extractFeatures(self,image, pose_in_image, xFeat=False, segment=False):
        mask, mask_3C = self.get_image_masks(image, pose_in_image,segment=segment)
        image = cv2.bitwise_and(mask_3C,image)
        
        if xFeat:
            output = self.xfeat.detectAndCompute(image,top_k = 4096)[0]
            kps = output['keypoints'].numpy()
            des = output['descriptors'].numpy()
            kps, des = self.filterFeatures(kps, des, mask, xFeat=xFeat)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Find the keypoints and descriptors with ORB
            kps, des = self.orb.detectAndCompute(gray, None)
            kps, des = self.filterFeatures(kps, des, mask, xFeat=xFeat)
            
        # print(kps.shape)
        # print(des.shape)
            
        return kps, des, mask,image
    
    def get_image_masks(self,image,pose_in_image,segment=False):
        tvec = pose_in_image['t']
        rvec = pose_in_image['R']
        mask,mask_3C = self.get_probe_mask(image, tvec, rvec, segment=segment)
        return mask, mask_3C
    
    def filterFeatures(self, kps, des, mask,xFeat=False):
        # Step 3: Filter Out Keypoints and Descriptors Using the Mask
        filtered_keypoints = []
        filtered_descriptors = []
        # print(kps)
        # print(des)
        
        for i, kp in enumerate(kps):
            # Check if the keypoint falls within the mask area
            if xFeat:
                x, y = kp[0], kp[1]
            else:
                x, y = kp.pt[0], kp.pt[1]
            if mask[int(y), int(x)] == 255:  # Keep keypoints that are outside the masked area
                filtered_keypoints.append([x,y])
                filtered_descriptors.append(des[i])
        
        # Convert the filtered descriptors to a NumPy array
        filtered_keypoints = np.array(filtered_keypoints)
        filtered_descriptors = np.array(filtered_descriptors)
        return filtered_keypoints,filtered_descriptors
    
    def filterKeyPoints(self,kps, mask,xFeat=False):
        # Step 3: Filter Out Keypoints and Descriptors Using the Mask
        filtered_keypoints = []
        
        
        for i, kp in enumerate(kps):
            # Check if the keypoint falls within the mask area
            if xFeat:
                x, y = kp[0], kp[1]
            else:
                x, y = kp.pt[0], kp.pt[1]
            if mask[int(y), int(x)] == 255:  # Keep keypoints that are outside the masked area
                filtered_keypoints.append([x,y])
                
        filtered_keypoints = np.array(filtered_keypoints)    
            
        return filtered_keypoints
    
    def filterMatchedKeyPoints(self,kp1, kp2, combined_mask):
        # Step 3: Filter Out Keypoints and Descriptors Using the Mask
        filtered_mkps1 = []
        filtered_mkps2 = []
        
        
        for i, kp in enumerate(kp1):
            # Check if the keypoint falls within the mask area
            x1,y1 = kp[0],kp[1]
            x2,y2 = kp2[i,0],kp2[i,1]
            
            if combined_mask[int(y1), int(x1)] == 255 and combined_mask[int(y2), int(x2)] == 255 :  # Keep keypoints that are outside the masked area
                filtered_mkps1.append([x1,y1])
                filtered_mkps2.append([x2,y2])
                
        filtered_mkps1 = np.array(filtered_mkps1) 
        filtered_mkps2 = np.array(filtered_mkps2)  
            
        return filtered_mkps1,filtered_mkps2
    
    def segment_probe(self,frame,points,labels):
        
        if self.predictor is None:
            self.load_SAM()
        
        self.predictor.set_image(frame)
        masks, _, _ = self.predictor.predict(point_coords=points,
                point_labels=labels,
                multimask_output=False,
            )
        
        mask = masks[0].astype(np.uint8) * 255
        return mask 
    def get_probe_mask(self, image, tvec, rvec, segment=False):
        # print(idx)
       
       
        probe_tip_offset = self.probe_tip_offsets[self.aruco_marker_ids[0]]
        shaft_length_above_marker = 70
        width = self.cube_size
        num_markers = self.num_markers
        probe_points, point_labels, probe_bb = U.getProbeBBAndPoints(probe_tip_offset, shaft_length_above_marker, width=width, num_markers=num_markers)
        cir_mask = U.create_circular_mask(self.frame_size[0], self.frame_size[1])
        # print(probe_points)
        # print(probe_bb)
        # print(tvec)
        # print(rvec)
        # print(self.dist_coeffs_for_projection)
        if segment:
            points,_ = cv2.projectPoints(probe_points, rvec, tvec, self.camera_matrix, self.dist_coeffs_for_projection)
            # Extract the image points
            image_points = points.reshape(-1, 2)  # Shape: (2, 2)
            probe_mask = self.segment_probe(image, image_points, point_labels)
            combined_mask = cv2.bitwise_and(cv2.bitwise_not(probe_mask),cir_mask)
            combined_mask_3C = cv2.merge([combined_mask, combined_mask, combined_mask])
            plt.imshow(combined_mask)
            return combined_mask,combined_mask_3C
        else:
            box_points,_ = cv2.projectPoints(probe_bb, rvec, tvec, self.camera_matrix, self.dist_coeffs_for_projection)
            
            # Extract the (x, y) coordinates of the projected points
            box_points_image = box_points.reshape(-1, 2)  # Shape: (4, 2)
            
            box_points_image = np.int0(box_points_image) 
            # print(box_points_image)
            
            # # Find the minimum and maximum coordinates to define the bounding box
            # x_min = np.min(projected_points_2d[:, 0])
            # y_min = np.min(projected_points_2d[:, 1])
            # x_max = np.max(projected_points_2d[:, 0])
            # y_max = np.max(projected_points_2d[:, 1])

            # # Define the bounding box
            # bounding_box = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
            # Create an empty mask of zeros (default background)
            probe_mask = np.zeros((self.frame_size[0], self.frame_size[1]), dtype=np.uint8)
            
            # Draw the rotated rectangle on the mask
            cv2.fillPoly(probe_mask, [box_points_image], 255)
            
            combined_mask = cv2.bitwise_and(cv2.bitwise_not(probe_mask),cir_mask)
            combined_mask_3C = cv2.merge([combined_mask, combined_mask, combined_mask])
            plt.imshow(combined_mask)
            plt.show()
            return combined_mask,combined_mask_3C
    def project_points_using_pca_and_compute_area(self,points):
        """
        Project 3D points to a 2D plane using PCA and compute the area.
        points should be a numpy array of shape (n, 3).
        """
        # Perform PCA to reduce dimensions from 3D to 2D
        pca = PCA(n_components=2)
        points_projected = pca.fit_transform(points)
        
        # Compute the area of the polygon formed by these projected points
        area = self.compute_polygon_area(points_projected)
        return area
    def compute_polygon_area(self,points):
        """
        Compute the area of a polygon given its vertices.
        Assumes that the points are [(x1, y1), (x2, y2), ..., (xn, yn)].
        Uses the shoelace formula.
        """
        x = points[:, 0]
        y = points[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def main():
    app = QApplication(sys.argv)
    window = CameraWidget()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
