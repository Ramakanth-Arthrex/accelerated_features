#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 13:22:20 2024

@author: ramakanth
"""

import numpy as np
import cv2
import random
import json
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QLineEdit, QGridLayout, QComboBox
from PyQt5.QtCore import pyqtSignal

def getIntrinsic(size,saline=False):
    r, c = size
    size4k = [2160, 3840]
    scale = r/size4k[0]
    #load directly
    # scopeNormalizedIntrinsics = np.load("/Users/ramakanth/Documents/ArthrexProjects/AREAS_Project/Arthrosocpy/localCam/scopeNormalizedIntrinsics.npy")
    # scopeDM = np.load("/Users/ramakanth/Documents/ArthrexProjects/AREAS_Project/Arthrosocpy/localCam/scopeDistCoeff.npy")
    
    # with open('//Users/ramakanth/Documents/ArthrexProjects/AREAS_Project/Arthrosocpy/localCam/accelerated_features/cal4k.json') as f:
    #     scopeIn = json.load(f)
    
    # CM = scopeIn["K"]
    # D = np.array(scopeIn["D"])
    
    # fy = CM[1][1]*size4k[0]* scale
    # fx = CM[0][0]*size4k[1]* scale
    
    # cy = (CM[1][2]*size4k[0] + scopeIn["C"][1]) *scale
    # cx = (CM[0][2]*size4k[1] + scopeIn["C"][0]) *scale
    
    with open('/Users/ramakanth/Documents/ArthrexProjects/AREAS_Project/Arthrosocpy/localCam/accelerated_features/scopecal_air_saline_placeholder.json') as f:
        scopeAllParameters = json.load(f)
    
    if saline:
        fx = scopeAllParameters['F_saline']*scale
        fy = fx
    else:
        fx = scopeAllParameters['F_air']*scale
        fy = fx
    
    cx = scopeAllParameters['CD_air'][1]*scale
    cy = scopeAllParameters['CD_air'][0]*scale
    
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
    
    D = np.array(scopeAllParameters['D_air'])
    
    return K,D
def roration_ambiguity(rvec1, rvec2):
    # Convert both to rotation matrices
    rotation_matrix1, _ = cv2.Rodrigues(rvec1)
    rotation_matrix2, _ = cv2.Rodrigues(rvec2)
    
    # Compute the difference (or similarity) between the two matrices
    diff_matrix = np.dot(rotation_matrix1.T, rotation_matrix2)
    angle_diff = np.arccos((np.trace(diff_matrix) - 1) / 2)
    
    return angle_diff
# def create_circular_mask(h, w, center=None, radius=None):
#     if center is None:
#         center = (w//2, h//2)
#     if radius is None:
#         radius = min(center[0], center[1], w-center[0], h-center[1])
#     Y, X = np.ogrid[:h, :w]
#     dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
#     mask = dist_from_center <= radius
#     return mask.astype(np.uint8)

def create_circular_mask(h, w, center=None, radius=None):
    # Step 1: Create a Circular Mask
    # Create a blank single-channel mask with the same dimensions as the image
    
    scale = 3840.0/w
    center=(int(1919.5//scale),int(1079.5//scale))
    radius=int(926//scale)
    circular_mask = np.zeros((h, w), dtype=np.uint8)

    # Draw a filled circle on the mask
    cv2.circle(circular_mask, center, radius, 255, thickness=-1)

    return circular_mask

def getProbeBBAndPoints(probe_tip_offset,shaft_length_above_marker,width=6,num_markers=3):
    # select two random points along probe shaft
    # Point 1: Random point below the ArUco marker along the shaft (between marker and probe tip)
    point_on_markers_offset = random.uniform(probe_tip_offset,num_markers*width)
    point_on_markers = np.array([0, point_on_markers_offset, 0])
    
    # Point 2: Random point above the ArUco marker along the shaft (between marker and 4cm above marker)
    above_point_offset = random.uniform(0, shaft_length_above_marker)
    point_above = np.array([0, (probe_tip_offset+(num_markers*width))+above_point_offset, 0])
    
    origin = np.array([0,0,0])
    
    
   
    
    
    # Bottom point of the probe (tip) relative to the ArUco marker
    probe_tip = np.array([0, 0, 0]).reshape(3, 1)
    
    # Top point of the probe (4 cm above the ArUco marker)
    probe_top = np.array([0, (probe_tip_offset+(num_markers*width))+shaft_length_above_marker,  0]).reshape(3, 1)
    
    len_probe = (probe_tip_offset+(num_markers*width))+shaft_length_above_marker
    
    # Define four corner points of the bounding box in the 3D space (assuming a rectangular cross-section for visualization)
    # For simplicity, assume the probe has a small width along the x-axis
    width = width*2  # 1 cm width
    
    # Define corner points in the probe object space
    probe_bb = np.array([
        [ - width / 2, 0,  0],   # Bottom-left
        [+ width / 2, 0,  0],   # Bottom-right
        [ + width / 2, len_probe,  0],   # Top-left
        [ - width / 2, len_probe,  0]   # Top-right
    ], dtype=np.float32)

    bg_offset = random.uniform(width, width+10)
    if random.uniform(0,10)>5:
        probe_bg = np.array([ + bg_offset, +bg_offset, 0] )
    else:
        probe_bg = np.array([ - bg_offset, +bg_offset,  0])
        
    probe_points = np.array([origin, point_on_markers , point_above, probe_bg])
    point_labels = np.array([1,1,1,0])

    return probe_points, point_labels,probe_bb


def getProbeBBAndPoints2(probe_tip_offset,shaft_length_above_marker,width=6):
    # select two random points along probe shaft
    # Point 1: Random point below the ArUco marker along the shaft (between marker and probe tip)
    below_point_offset = random.uniform(0, probe_tip_offset)
    point_below = np.array([0, -below_point_offset, - width / 2])
    
    # Point 2: Random point above the ArUco marker along the shaft (between marker and 4cm above marker)
    above_point_offset = random.uniform(0, shaft_length_above_marker)
    point_above = np.array([0, above_point_offset, - width / 2])
    
    origin = np.array([0,0,-width/2])
    
    
   
    
    
    # Bottom point of the probe (tip) relative to the ArUco marker
    probe_tip = np.array([0, -probe_tip_offset, - width / 2]).reshape(3, 1)
    
    # Top point of the probe (4 cm above the ArUco marker)
    probe_top = np.array([0, shaft_length_above_marker,  - width / 2]).reshape(3, 1)
    
    # Define four corner points of the bounding box in the 3D space (assuming a rectangular cross-section for visualization)
    # For simplicity, assume the probe has a small width along the x-axis
    width = width  # 1 cm width
    
    # Define corner points in the probe object space
    probe_bb = np.array([
        [ - width / 2, -probe_tip_offset,  - width / 2],   # Bottom-left
        [+ width / 2, -probe_tip_offset,  - width / 2],   # Bottom-right
        [ + width / 2, shaft_length_above_marker,  - width / 2],   # Top-left
        [ - width / 2, shaft_length_above_marker,  - width / 2]   # Top-right
    ], dtype=np.float32)

    bg_offset = random.uniform(width, width+10)
    if random.uniform(0,10)>5:
        probe_bg = np.array([ + bg_offset, -probe_tip_offset+bg_offset, - width / 2] )
    else:
        probe_bg = np.array([ - bg_offset, -probe_tip_offset+bg_offset,  - width / 2])
        
    probe_points = np.array([origin, point_below, point_above,probe_bg])
    point_labels = np.array([1,1,1,0])

    return probe_points, point_labels,probe_bb

class SettingsWindow(QWidget):
    
    # Defining a custom signal that emits a dict
    settings_updated = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Settings")
        layout = QGridLayout()

        # Probe tip position y axis- offsets with resspect to aruco marker cube
        layout.addWidget(QLabel('Probe OffSetY 1:'), 0, 0)
        self.probe_tip1 = QLineEdit(self)
        layout.addWidget(self.probe_tip1, 0, 1)

        layout.addWidget(QLabel('Probe OffSetY 2'), 1, 0)
        self.probe_tip2 = QLineEdit(self)
        layout.addWidget(self.probe_tip2, 1, 1)

        layout.addWidget(QLabel('Probe OffSetY 3:'), 2, 0)
        self.probe_tip3 = QLineEdit(self)
        layout.addWidget(self.probe_tip3, 2, 1)
        
        # aurco marker configuration
        layout.addWidget(QLabel('Number of Markers:'), 3, 0)
        self.num_markers = QLineEdit(self)
        layout.addWidget(self.num_markers, 3, 1)

        layout.addWidget(QLabel('Marker Size:'), 4, 0)
        self.marker_size = QLineEdit(self)
        layout.addWidget(self.marker_size, 4, 1)

        # Video frame size
        layout.addWidget(QLabel('Frame Width:'), 5, 0)
        self.frame_width = QLineEdit(self)
        layout.addWidget(self.frame_width, 5, 1)

        layout.addWidget(QLabel('Frame Height:'), 6, 0)
        self.frame_height = QLineEdit(self)
        layout.addWidget(self.frame_height, 6, 1)

        # Feature detection method selection
        layout.addWidget(QLabel('Feature Method:'), 7, 0)
        self.feature_method = QComboBox(self)
        self.feature_method.addItems(['xFeat', 'ORB'])
        layout.addWidget(self.feature_method, 7, 1)
        
        # Feature detection method selection
        layout.addWidget(QLabel('Segement Probe:'), 8, 0)
        self.segment_method = QComboBox(self)
        self.segment_method.addItems(['SAM', 'BB'])
        layout.addWidget(self.segment_method, 8, 1)
        
        # Confirm button
        self.confirm_button = QPushButton('Confirm', self)
        self.confirm_button.clicked.connect(self.emitSettings)
        layout.addWidget(self.confirm_button, 9, 0, 1, 2)  # Span across two columns

        

        self.setLayout(layout)
        
    
    def emitSettings(self):
        settings = {
            'aruco_configuration': [self.num_markers,self.marker_size], #[int(self.num_markers.text()) if self.num_markers else 3 ,int(self.marker_size.text()) if self.marker_size else 3],
            'probe_tip_offsets': [self.probe_x.text(),self.probe_y.text(),self.probe_z.text()],
            'frame_size': [self.frame_height.text(),self.frame_width.text()],
            'feature_method': self.feature_method.currentText(),
            'segement_method':self.segment_method.currentText()
        }
        self.settings_updated.emit(settings)
        self.close()


