# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Deep_FakeJdcrxc.ui'
##
## Created by: Qt User Interface Compiler version 6.9.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFormLayout,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QMainWindow, QProgressBar, QPushButton, QScrollBar,
    QSizePolicy, QSlider, QSpacerItem, QStatusBar,
    QTabWidget, QVBoxLayout, QWidget)

class Ui_DeepFakeMainWindow(object):
    def setupUi(self, DeepFakeMainWindow):
        if not DeepFakeMainWindow.objectName():
            DeepFakeMainWindow.setObjectName(u"DeepFakeMainWindow")
        DeepFakeMainWindow.resize(1118, 1049)
        DeepFakeMainWindow.setStyleSheet(u"")
        self.centralwidget = QWidget(DeepFakeMainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.input_tab = QWidget()
        self.input_tab.setObjectName(u"input_tab")
        self.verticalLayout_2 = QVBoxLayout(self.input_tab)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.groupBox = QGroupBox(self.input_tab)
        self.groupBox.setObjectName(u"groupBox")
        self.gridLayout = QGridLayout(self.groupBox)
        self.gridLayout.setObjectName(u"gridLayout")
        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.video_label = QLabel(self.groupBox)
        self.video_label.setObjectName(u"video_label")
        self.video_label.setMinimumSize(QSize(0, 40))
        self.video_label.setStyleSheet(u"QLabel { border: 2px dashed #666; padding: 10px; }")

        self.gridLayout.addWidget(self.video_label, 0, 1, 1, 1)

        self.select_video_btn = QPushButton(self.groupBox)
        self.select_video_btn.setObjectName(u"select_video_btn")

        self.gridLayout.addWidget(self.select_video_btn, 0, 2, 1, 1)

        self.label_2 = QLabel(self.groupBox)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)

        self.image_label = QLabel(self.groupBox)
        self.image_label.setObjectName(u"image_label")
        self.image_label.setMinimumSize(QSize(0, 40))
        self.image_label.setStyleSheet(u"QLabel { border: 2px dashed #666; padding: 10px; }")

        self.gridLayout.addWidget(self.image_label, 1, 1, 1, 1)

        self.select_image_btn = QPushButton(self.groupBox)
        self.select_image_btn.setObjectName(u"select_image_btn")

        self.gridLayout.addWidget(self.select_image_btn, 1, 2, 1, 1)


        self.verticalLayout_2.addWidget(self.groupBox)

        self.groupBox_2 = QGroupBox(self.input_tab)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.horizontalLayout = QHBoxLayout(self.groupBox_2)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.video_preview = QLabel(self.groupBox_2)
        self.video_preview.setObjectName(u"video_preview")
        self.video_preview.setMinimumSize(QSize(400, 300))
        self.video_preview.setStyleSheet(u"QLabel { border: 1px solid #666; background-color: #000; }")
        self.video_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout.addWidget(self.video_preview)

        self.image_preview = QLabel(self.groupBox_2)
        self.image_preview.setObjectName(u"image_preview")
        self.image_preview.setMinimumSize(QSize(400, 300))
        self.image_preview.setStyleSheet(u"QLabel { border: 1px solid #666; background-color: #000; }")
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout.addWidget(self.image_preview)


        self.verticalLayout_2.addWidget(self.groupBox_2)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer)

        self.next_btn_1 = QPushButton(self.input_tab)
        self.next_btn_1.setObjectName(u"next_btn_1")
        font = QFont()
        font.setFamilies([u"ROG Fonts STRIX SCAR"])
        font.setPointSize(11)
        font.setBold(False)
        self.next_btn_1.setFont(font)

        self.horizontalLayout_2.addWidget(self.next_btn_1)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_2)


        self.verticalLayout_2.addLayout(self.horizontalLayout_2)

        self.tabWidget.addTab(self.input_tab, "")
        self.detection_tab = QWidget()
        self.detection_tab.setObjectName(u"detection_tab")
        self.verticalLayout_3 = QVBoxLayout(self.detection_tab)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.groupBox_3 = QGroupBox(self.detection_tab)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.formLayout = QFormLayout(self.groupBox_3)
        self.formLayout.setObjectName(u"formLayout")
        self.label_3 = QLabel(self.groupBox_3)
        self.label_3.setObjectName(u"label_3")

        self.formLayout.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_3)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.conf_threshold_slider = QSlider(self.groupBox_3)
        self.conf_threshold_slider.setObjectName(u"conf_threshold_slider")
        self.conf_threshold_slider.setMinimum(10)
        self.conf_threshold_slider.setMaximum(90)
        self.conf_threshold_slider.setValue(50)
        self.conf_threshold_slider.setOrientation(Qt.Orientation.Horizontal)

        self.horizontalLayout_3.addWidget(self.conf_threshold_slider)

        self.conf_threshold_label = QLabel(self.groupBox_3)
        self.conf_threshold_label.setObjectName(u"conf_threshold_label")

        self.horizontalLayout_3.addWidget(self.conf_threshold_label)


        self.formLayout.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_3)

        self.label_4 = QLabel(self.groupBox_3)
        self.label_4.setObjectName(u"label_4")

        self.formLayout.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_4)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.nms_threshold_slider = QSlider(self.groupBox_3)
        self.nms_threshold_slider.setObjectName(u"nms_threshold_slider")
        self.nms_threshold_slider.setMinimum(10)
        self.nms_threshold_slider.setMaximum(90)
        self.nms_threshold_slider.setValue(40)
        self.nms_threshold_slider.setOrientation(Qt.Orientation.Horizontal)

        self.horizontalLayout_4.addWidget(self.nms_threshold_slider)

        self.nms_threshold_label = QLabel(self.groupBox_3)
        self.nms_threshold_label.setObjectName(u"nms_threshold_label")

        self.horizontalLayout_4.addWidget(self.nms_threshold_label)


        self.formLayout.setLayout(1, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_4)

        self.detect_btn = QPushButton(self.groupBox_3)
        self.detect_btn.setObjectName(u"detect_btn")

        self.formLayout.setWidget(2, QFormLayout.ItemRole.SpanningRole, self.detect_btn)


        self.verticalLayout_3.addWidget(self.groupBox_3)

        self.detection_preview = QLabel(self.detection_tab)
        self.detection_preview.setObjectName(u"detection_preview")
        self.detection_preview.setMinimumSize(QSize(800, 600))
        self.detection_preview.setStyleSheet(u"QLabel { border: 1px solid #666; background-color: #000; }")
        self.detection_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_3.addWidget(self.detection_preview)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.prev_btn_2 = QPushButton(self.detection_tab)
        self.prev_btn_2.setObjectName(u"prev_btn_2")

        self.horizontalLayout_5.addWidget(self.prev_btn_2)

        self.next_btn_2 = QPushButton(self.detection_tab)
        self.next_btn_2.setObjectName(u"next_btn_2")

        self.horizontalLayout_5.addWidget(self.next_btn_2)


        self.verticalLayout_3.addLayout(self.horizontalLayout_5)

        self.tabWidget.addTab(self.detection_tab, "")
        self.landmarks_tab = QWidget()
        self.landmarks_tab.setObjectName(u"landmarks_tab")
        self.verticalLayout_4 = QVBoxLayout(self.landmarks_tab)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.groupBox_4 = QGroupBox(self.landmarks_tab)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.formLayout_2 = QFormLayout(self.groupBox_4)
        self.formLayout_2.setObjectName(u"formLayout_2")
        self.label_5 = QLabel(self.groupBox_4)
        self.label_5.setObjectName(u"label_5")

        self.formLayout_2.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_5)

        self.landmark_model_combo = QComboBox(self.groupBox_4)
        self.landmark_model_combo.addItem("")
        self.landmark_model_combo.addItem("")
        self.landmark_model_combo.addItem("")
        self.landmark_model_combo.setObjectName(u"landmark_model_combo")

        self.formLayout_2.setWidget(0, QFormLayout.ItemRole.FieldRole, self.landmark_model_combo)

        self.label_6 = QLabel(self.groupBox_4)
        self.label_6.setObjectName(u"label_6")

        self.formLayout_2.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_6)

        self.landmark_points_combo = QComboBox(self.groupBox_4)
        self.landmark_points_combo.addItem("")
        self.landmark_points_combo.addItem("")
        self.landmark_points_combo.addItem("")
        self.landmark_points_combo.setObjectName(u"landmark_points_combo")

        self.formLayout_2.setWidget(1, QFormLayout.ItemRole.FieldRole, self.landmark_points_combo)

        self.detect_landmarks_btn = QPushButton(self.groupBox_4)
        self.detect_landmarks_btn.setObjectName(u"detect_landmarks_btn")

        self.formLayout_2.setWidget(2, QFormLayout.ItemRole.SpanningRole, self.detect_landmarks_btn)


        self.verticalLayout_4.addWidget(self.groupBox_4)

        self.landmarks_preview = QLabel(self.landmarks_tab)
        self.landmarks_preview.setObjectName(u"landmarks_preview")
        self.landmarks_preview.setMinimumSize(QSize(800, 600))
        self.landmarks_preview.setStyleSheet(u"QLabel { border: 1px solid #666; background-color: #000; }")
        self.landmarks_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_4.addWidget(self.landmarks_preview)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.prev_btn_3 = QPushButton(self.landmarks_tab)
        self.prev_btn_3.setObjectName(u"prev_btn_3")

        self.horizontalLayout_6.addWidget(self.prev_btn_3)

        self.next_btn_3 = QPushButton(self.landmarks_tab)
        self.next_btn_3.setObjectName(u"next_btn_3")

        self.horizontalLayout_6.addWidget(self.next_btn_3)


        self.verticalLayout_4.addLayout(self.horizontalLayout_6)

        self.tabWidget.addTab(self.landmarks_tab, "")
        self.alignment_tab = QWidget()
        self.alignment_tab.setObjectName(u"alignment_tab")
        self.verticalLayout_5 = QVBoxLayout(self.alignment_tab)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.groupBox_5 = QGroupBox(self.alignment_tab)
        self.groupBox_5.setObjectName(u"groupBox_5")
        self.formLayout_3 = QFormLayout(self.groupBox_5)
        self.formLayout_3.setObjectName(u"formLayout_3")
        self.label_7 = QLabel(self.groupBox_5)
        self.label_7.setObjectName(u"label_7")

        self.formLayout_3.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_7)

        self.alignment_method_combo = QComboBox(self.groupBox_5)
        self.alignment_method_combo.addItem("")
        self.alignment_method_combo.addItem("")
        self.alignment_method_combo.addItem("")
        self.alignment_method_combo.setObjectName(u"alignment_method_combo")

        self.formLayout_3.setWidget(0, QFormLayout.ItemRole.FieldRole, self.alignment_method_combo)

        self.label_8 = QLabel(self.groupBox_5)
        self.label_8.setObjectName(u"label_8")

        self.formLayout_3.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_8)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.face_size_slider = QSlider(self.groupBox_5)
        self.face_size_slider.setObjectName(u"face_size_slider")
        self.face_size_slider.setMinimum(64)
        self.face_size_slider.setMaximum(512)
        self.face_size_slider.setValue(256)
        self.face_size_slider.setOrientation(Qt.Orientation.Horizontal)

        self.horizontalLayout_7.addWidget(self.face_size_slider)

        self.face_size_label = QLabel(self.groupBox_5)
        self.face_size_label.setObjectName(u"face_size_label")

        self.horizontalLayout_7.addWidget(self.face_size_label)


        self.formLayout_3.setLayout(1, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_7)

        self.align_faces_btn = QPushButton(self.groupBox_5)
        self.align_faces_btn.setObjectName(u"align_faces_btn")

        self.formLayout_3.setWidget(2, QFormLayout.ItemRole.SpanningRole, self.align_faces_btn)


        self.verticalLayout_5.addWidget(self.groupBox_5)

        self.alignment_preview = QLabel(self.alignment_tab)
        self.alignment_preview.setObjectName(u"alignment_preview")
        self.alignment_preview.setMinimumSize(QSize(800, 600))
        self.alignment_preview.setStyleSheet(u"QLabel { border: 1px solid #666; background-color: #000; }")
        self.alignment_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_5.addWidget(self.alignment_preview)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.prev_btn_4 = QPushButton(self.alignment_tab)
        self.prev_btn_4.setObjectName(u"prev_btn_4")

        self.horizontalLayout_8.addWidget(self.prev_btn_4)

        self.next_btn_4 = QPushButton(self.alignment_tab)
        self.next_btn_4.setObjectName(u"next_btn_4")

        self.horizontalLayout_8.addWidget(self.next_btn_4)


        self.verticalLayout_5.addLayout(self.horizontalLayout_8)

        self.tabWidget.addTab(self.alignment_tab, "")
        self.recognition_tab = QWidget()
        self.recognition_tab.setObjectName(u"recognition_tab")
        self.verticalLayout_6 = QVBoxLayout(self.recognition_tab)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.groupBox_6 = QGroupBox(self.recognition_tab)
        self.groupBox_6.setObjectName(u"groupBox_6")
        self.formLayout_4 = QFormLayout(self.groupBox_6)
        self.formLayout_4.setObjectName(u"formLayout_4")
        self.label_9 = QLabel(self.groupBox_6)
        self.label_9.setObjectName(u"label_9")

        self.formLayout_4.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_9)

        self.recognition_model_combo = QComboBox(self.groupBox_6)
        self.recognition_model_combo.addItem("")
        self.recognition_model_combo.addItem("")
        self.recognition_model_combo.addItem("")
        self.recognition_model_combo.setObjectName(u"recognition_model_combo")

        self.formLayout_4.setWidget(0, QFormLayout.ItemRole.FieldRole, self.recognition_model_combo)

        self.label_10 = QLabel(self.groupBox_6)
        self.label_10.setObjectName(u"label_10")

        self.formLayout_4.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_10)

        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.similarity_threshold_slider = QSlider(self.groupBox_6)
        self.similarity_threshold_slider.setObjectName(u"similarity_threshold_slider")
        self.similarity_threshold_slider.setMinimum(50)
        self.similarity_threshold_slider.setMaximum(99)
        self.similarity_threshold_slider.setValue(80)
        self.similarity_threshold_slider.setOrientation(Qt.Orientation.Horizontal)

        self.horizontalLayout_9.addWidget(self.similarity_threshold_slider)

        self.similarity_threshold_label = QLabel(self.groupBox_6)
        self.similarity_threshold_label.setObjectName(u"similarity_threshold_label")

        self.horizontalLayout_9.addWidget(self.similarity_threshold_label)


        self.formLayout_4.setLayout(1, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_9)

        self.recognize_faces_btn = QPushButton(self.groupBox_6)
        self.recognize_faces_btn.setObjectName(u"recognize_faces_btn")

        self.formLayout_4.setWidget(2, QFormLayout.ItemRole.SpanningRole, self.recognize_faces_btn)


        self.verticalLayout_6.addWidget(self.groupBox_6)

        self.recognition_preview = QLabel(self.recognition_tab)
        self.recognition_preview.setObjectName(u"recognition_preview")
        self.recognition_preview.setMinimumSize(QSize(800, 600))
        self.recognition_preview.setStyleSheet(u"QLabel { border: 1px solid #666; background-color: #000; }")
        self.recognition_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_6.addWidget(self.recognition_preview)

        self.horizontalLayout_10 = QHBoxLayout()
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.prev_btn_5 = QPushButton(self.recognition_tab)
        self.prev_btn_5.setObjectName(u"prev_btn_5")

        self.horizontalLayout_10.addWidget(self.prev_btn_5)

        self.next_btn_5 = QPushButton(self.recognition_tab)
        self.next_btn_5.setObjectName(u"next_btn_5")

        self.horizontalLayout_10.addWidget(self.next_btn_5)


        self.verticalLayout_6.addLayout(self.horizontalLayout_10)

        self.tabWidget.addTab(self.recognition_tab, "")
        self.segmentation_tab = QWidget()
        self.segmentation_tab.setObjectName(u"segmentation_tab")
        self.verticalLayout_7 = QVBoxLayout(self.segmentation_tab)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.groupBox_7 = QGroupBox(self.segmentation_tab)
        self.groupBox_7.setObjectName(u"groupBox_7")
        self.formLayout_5 = QFormLayout(self.groupBox_7)
        self.formLayout_5.setObjectName(u"formLayout_5")
        self.label_11 = QLabel(self.groupBox_7)
        self.label_11.setObjectName(u"label_11")

        self.formLayout_5.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_11)

        self.segmentation_model_combo = QComboBox(self.groupBox_7)
        self.segmentation_model_combo.addItem("")
        self.segmentation_model_combo.addItem("")
        self.segmentation_model_combo.addItem("")
        self.segmentation_model_combo.setObjectName(u"segmentation_model_combo")

        self.formLayout_5.setWidget(0, QFormLayout.ItemRole.FieldRole, self.segmentation_model_combo)

        self.label_12 = QLabel(self.groupBox_7)
        self.label_12.setObjectName(u"label_12")

        self.formLayout_5.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_12)

        self.horizontalLayout_11 = QHBoxLayout()
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.mask_blur_slider = QSlider(self.groupBox_7)
        self.mask_blur_slider.setObjectName(u"mask_blur_slider")
        self.mask_blur_slider.setMinimum(0)
        self.mask_blur_slider.setMaximum(20)
        self.mask_blur_slider.setValue(5)
        self.mask_blur_slider.setOrientation(Qt.Orientation.Horizontal)

        self.horizontalLayout_11.addWidget(self.mask_blur_slider)

        self.mask_blur_label = QLabel(self.groupBox_7)
        self.mask_blur_label.setObjectName(u"mask_blur_label")

        self.horizontalLayout_11.addWidget(self.mask_blur_label)


        self.formLayout_5.setLayout(1, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_11)

        self.segment_faces_btn = QPushButton(self.groupBox_7)
        self.segment_faces_btn.setObjectName(u"segment_faces_btn")

        self.formLayout_5.setWidget(2, QFormLayout.ItemRole.SpanningRole, self.segment_faces_btn)


        self.verticalLayout_7.addWidget(self.groupBox_7)

        self.segmentation_preview = QLabel(self.segmentation_tab)
        self.segmentation_preview.setObjectName(u"segmentation_preview")
        self.segmentation_preview.setMinimumSize(QSize(800, 600))
        self.segmentation_preview.setStyleSheet(u"QLabel { border: 1px solid #666; background-color: #000; }")
        self.segmentation_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_7.addWidget(self.segmentation_preview)

        self.horizontalLayout_12 = QHBoxLayout()
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.prev_btn_6 = QPushButton(self.segmentation_tab)
        self.prev_btn_6.setObjectName(u"prev_btn_6")

        self.horizontalLayout_12.addWidget(self.prev_btn_6)

        self.next_btn_6 = QPushButton(self.segmentation_tab)
        self.next_btn_6.setObjectName(u"next_btn_6")

        self.horizontalLayout_12.addWidget(self.next_btn_6)


        self.verticalLayout_7.addLayout(self.horizontalLayout_12)

        self.tabWidget.addTab(self.segmentation_tab, "")
        self.swapping_tab = QWidget()
        self.swapping_tab.setObjectName(u"swapping_tab")
        self.verticalLayout_8 = QVBoxLayout(self.swapping_tab)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.groupBox_8 = QGroupBox(self.swapping_tab)
        self.groupBox_8.setObjectName(u"groupBox_8")
        self.formLayout_6 = QFormLayout(self.groupBox_8)
        self.formLayout_6.setObjectName(u"formLayout_6")
        self.label_13 = QLabel(self.groupBox_8)
        self.label_13.setObjectName(u"label_13")

        self.formLayout_6.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_13)

        self.swap_model_combo = QComboBox(self.groupBox_8)
        self.swap_model_combo.addItem("")
        self.swap_model_combo.addItem("")
        self.swap_model_combo.addItem("")
        self.swap_model_combo.setObjectName(u"swap_model_combo")

        self.formLayout_6.setWidget(0, QFormLayout.ItemRole.FieldRole, self.swap_model_combo)

        self.label_14 = QLabel(self.groupBox_8)
        self.label_14.setObjectName(u"label_14")

        self.formLayout_6.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_14)

        self.horizontalLayout_13 = QHBoxLayout()
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.blend_ratio_slider = QSlider(self.groupBox_8)
        self.blend_ratio_slider.setObjectName(u"blend_ratio_slider")
        self.blend_ratio_slider.setMinimum(0)
        self.blend_ratio_slider.setMaximum(100)
        self.blend_ratio_slider.setValue(80)
        self.blend_ratio_slider.setOrientation(Qt.Orientation.Horizontal)

        self.horizontalLayout_13.addWidget(self.blend_ratio_slider)

        self.blend_ratio_label = QLabel(self.groupBox_8)
        self.blend_ratio_label.setObjectName(u"blend_ratio_label")

        self.horizontalLayout_13.addWidget(self.blend_ratio_label)


        self.formLayout_6.setLayout(1, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_13)

        self.label_15 = QLabel(self.groupBox_8)
        self.label_15.setObjectName(u"label_15")

        self.formLayout_6.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_15)

        self.color_correction_check = QCheckBox(self.groupBox_8)
        self.color_correction_check.setObjectName(u"color_correction_check")
        self.color_correction_check.setChecked(True)

        self.formLayout_6.setWidget(2, QFormLayout.ItemRole.FieldRole, self.color_correction_check)

        self.start_swap_btn = QPushButton(self.groupBox_8)
        self.start_swap_btn.setObjectName(u"start_swap_btn")

        self.formLayout_6.setWidget(3, QFormLayout.ItemRole.SpanningRole, self.start_swap_btn)


        self.verticalLayout_8.addWidget(self.groupBox_8)

        self.swapping_preview = QLabel(self.swapping_tab)
        self.swapping_preview.setObjectName(u"swapping_preview")
        self.swapping_preview.setMinimumSize(QSize(800, 600))
        self.swapping_preview.setStyleSheet(u"QLabel { border: 1px solid #666; background-color: #000; }")
        self.swapping_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_8.addWidget(self.swapping_preview)

        self.horizontalLayout_14 = QHBoxLayout()
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.prev_btn_7 = QPushButton(self.swapping_tab)
        self.prev_btn_7.setObjectName(u"prev_btn_7")

        self.horizontalLayout_14.addWidget(self.prev_btn_7)

        self.next_btn_7 = QPushButton(self.swapping_tab)
        self.next_btn_7.setObjectName(u"next_btn_7")

        self.horizontalLayout_14.addWidget(self.next_btn_7)


        self.verticalLayout_8.addLayout(self.horizontalLayout_14)

        self.tabWidget.addTab(self.swapping_tab, "")
        self.export_tab = QWidget()
        self.export_tab.setObjectName(u"export_tab")
        self.verticalLayout_9 = QVBoxLayout(self.export_tab)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.groupBox_9 = QGroupBox(self.export_tab)
        self.groupBox_9.setObjectName(u"groupBox_9")
        self.formLayout_7 = QFormLayout(self.groupBox_9)
        self.formLayout_7.setObjectName(u"formLayout_7")
        self.label_16 = QLabel(self.groupBox_9)
        self.label_16.setObjectName(u"label_16")

        self.formLayout_7.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_16)

        self.output_format_combo = QComboBox(self.groupBox_9)
        self.output_format_combo.addItem("")
        self.output_format_combo.addItem("")
        self.output_format_combo.addItem("")
        self.output_format_combo.addItem("")
        self.output_format_combo.setObjectName(u"output_format_combo")

        self.formLayout_7.setWidget(0, QFormLayout.ItemRole.FieldRole, self.output_format_combo)

        self.label_17 = QLabel(self.groupBox_9)
        self.label_17.setObjectName(u"label_17")

        self.formLayout_7.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_17)

        self.horizontalLayout_15 = QHBoxLayout()
        self.horizontalLayout_15.setObjectName(u"horizontalLayout_15")
        self.quality_slider = QSlider(self.groupBox_9)
        self.quality_slider.setObjectName(u"quality_slider")
        self.quality_slider.setMinimum(1)
        self.quality_slider.setMaximum(10)
        self.quality_slider.setValue(8)
        self.quality_slider.setOrientation(Qt.Orientation.Horizontal)

        self.horizontalLayout_15.addWidget(self.quality_slider)

        self.quality_label = QLabel(self.groupBox_9)
        self.quality_label.setObjectName(u"quality_label")

        self.horizontalLayout_15.addWidget(self.quality_label)


        self.formLayout_7.setLayout(1, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_15)

        self.label_18 = QLabel(self.groupBox_9)
        self.label_18.setObjectName(u"label_18")

        self.formLayout_7.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_18)

        self.fps_combo = QComboBox(self.groupBox_9)
        self.fps_combo.addItem("")
        self.fps_combo.addItem("")
        self.fps_combo.addItem("")
        self.fps_combo.addItem("")
        self.fps_combo.setObjectName(u"fps_combo")

        self.formLayout_7.setWidget(2, QFormLayout.ItemRole.FieldRole, self.fps_combo)

        self.label_19 = QLabel(self.groupBox_9)
        self.label_19.setObjectName(u"label_19")

        self.formLayout_7.setWidget(3, QFormLayout.ItemRole.LabelRole, self.label_19)

        self.horizontalLayout_16 = QHBoxLayout()
        self.horizontalLayout_16.setObjectName(u"horizontalLayout_16")
        self.output_path_label = QLabel(self.groupBox_9)
        self.output_path_label.setObjectName(u"output_path_label")

        self.horizontalLayout_16.addWidget(self.output_path_label)

        self.select_output_btn = QPushButton(self.groupBox_9)
        self.select_output_btn.setObjectName(u"select_output_btn")

        self.horizontalLayout_16.addWidget(self.select_output_btn)


        self.formLayout_7.setLayout(3, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_16)

        self.export_btn = QPushButton(self.groupBox_9)
        self.export_btn.setObjectName(u"export_btn")

        self.formLayout_7.setWidget(4, QFormLayout.ItemRole.SpanningRole, self.export_btn)


        self.verticalLayout_9.addWidget(self.groupBox_9)

        self.export_progress = QProgressBar(self.export_tab)
        self.export_progress.setObjectName(u"export_progress")

        self.verticalLayout_9.addWidget(self.export_progress)

        self.export_preview = QLabel(self.export_tab)
        self.export_preview.setObjectName(u"export_preview")
        self.export_preview.setMinimumSize(QSize(800, 600))
        self.export_preview.setStyleSheet(u"QLabel { border: 1px solid #666; background-color: #000; }")
        self.export_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_9.addWidget(self.export_preview)

        self.horizontalLayout_17 = QHBoxLayout()
        self.horizontalLayout_17.setObjectName(u"horizontalLayout_17")
        self.prev_btn_8 = QPushButton(self.export_tab)
        self.prev_btn_8.setObjectName(u"prev_btn_8")

        self.horizontalLayout_17.addWidget(self.prev_btn_8)

        self.horizontalSpacer_8 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_17.addItem(self.horizontalSpacer_8)


        self.verticalLayout_9.addLayout(self.horizontalLayout_17)

        self.tabWidget.addTab(self.export_tab, "")

        self.verticalLayout.addWidget(self.tabWidget)

        self.verticalLayout_10 = QVBoxLayout()
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.horizontalLayout_18 = QHBoxLayout()
        self.horizontalLayout_18.setObjectName(u"horizontalLayout_18")
        self.play_pause_btn = QPushButton(self.centralwidget)
        self.play_pause_btn.setObjectName(u"play_pause_btn")
        self.play_pause_btn.setMaximumSize(QSize(40, 40))

        self.horizontalLayout_18.addWidget(self.play_pause_btn)

        self.stop_btn = QPushButton(self.centralwidget)
        self.stop_btn.setObjectName(u"stop_btn")
        self.stop_btn.setMaximumSize(QSize(40, 40))

        self.horizontalLayout_18.addWidget(self.stop_btn)

        self.frame_slider = QSlider(self.centralwidget)
        self.frame_slider.setObjectName(u"frame_slider")
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)
        self.frame_slider.setOrientation(Qt.Orientation.Horizontal)

        self.horizontalLayout_18.addWidget(self.frame_slider)

        self.time_label = QLabel(self.centralwidget)
        self.time_label.setObjectName(u"time_label")

        self.horizontalLayout_18.addWidget(self.time_label)

        self.frame_label = QLabel(self.centralwidget)
        self.frame_label.setObjectName(u"frame_label")

        self.horizontalLayout_18.addWidget(self.frame_label)


        self.verticalLayout_10.addLayout(self.horizontalLayout_18)

        self.horizontalLayout_19 = QHBoxLayout()
        self.horizontalLayout_19.setObjectName(u"horizontalLayout_19")
        self.label_20 = QLabel(self.centralwidget)
        self.label_20.setObjectName(u"label_20")

        self.horizontalLayout_19.addWidget(self.label_20)

        self.timeline_scroll = QScrollBar(self.centralwidget)
        self.timeline_scroll.setObjectName(u"timeline_scroll")
        self.timeline_scroll.setMinimum(0)
        self.timeline_scroll.setMaximum(100)
        self.timeline_scroll.setOrientation(Qt.Orientation.Horizontal)

        self.horizontalLayout_19.addWidget(self.timeline_scroll)

        self.label_21 = QLabel(self.centralwidget)
        self.label_21.setObjectName(u"label_21")

        self.horizontalLayout_19.addWidget(self.label_21)

        self.zoom_slider = QSlider(self.centralwidget)
        self.zoom_slider.setObjectName(u"zoom_slider")
        self.zoom_slider.setMaximumSize(QSize(100, 16777215))
        self.zoom_slider.setMinimum(10)
        self.zoom_slider.setMaximum(200)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setOrientation(Qt.Orientation.Horizontal)

        self.horizontalLayout_19.addWidget(self.zoom_slider)


        self.verticalLayout_10.addLayout(self.horizontalLayout_19)


        self.verticalLayout.addLayout(self.verticalLayout_10)

        DeepFakeMainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(DeepFakeMainWindow)
        self.statusbar.setObjectName(u"statusbar")
        self.statusbar.setStyleSheet(u"background-color: #404040; color: #ffffff;")
        DeepFakeMainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(DeepFakeMainWindow)
        self.conf_threshold_slider.valueChanged.connect(self.conf_threshold_label.setNum)
        self.nms_threshold_slider.valueChanged.connect(self.nms_threshold_label.setNum)
        self.face_size_slider.valueChanged.connect(self.face_size_label.setNum)
        self.similarity_threshold_slider.valueChanged.connect(self.similarity_threshold_label.setNum)
        self.mask_blur_slider.valueChanged.connect(self.mask_blur_label.setNum)
        self.blend_ratio_slider.valueChanged.connect(self.blend_ratio_label.setNum)
        self.quality_slider.valueChanged.connect(self.quality_label.setNum)

        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(DeepFakeMainWindow)
    # setupUi

    def retranslateUi(self, DeepFakeMainWindow):
        DeepFakeMainWindow.setWindowTitle(QCoreApplication.translate("DeepFakeMainWindow", u"Deep Fake Studio Pro", None))
        self.groupBox.setTitle(QCoreApplication.translate("DeepFakeMainWindow", u"Input Files", None))
        self.label.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Target Video:", None))
        self.video_label.setText(QCoreApplication.translate("DeepFakeMainWindow", u"No video selected", None))
        self.select_video_btn.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Select Video", None))
        self.label_2.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Source Image:", None))
        self.image_label.setText(QCoreApplication.translate("DeepFakeMainWindow", u"No image selected", None))
        self.select_image_btn.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Select Source Image", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("DeepFakeMainWindow", u"Preview", None))
        self.video_preview.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Video Preview", None))
        self.image_preview.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Image Preview", None))
        self.next_btn_1.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Next", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.input_tab), QCoreApplication.translate("DeepFakeMainWindow", u"1. Input", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("DeepFakeMainWindow", u"Detection Settings", None))
        self.label_3.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Confidence Threshold:", None))
        self.conf_threshold_label.setText(QCoreApplication.translate("DeepFakeMainWindow", u"0.50", None))
        self.label_4.setText(QCoreApplication.translate("DeepFakeMainWindow", u"NMS Threshold:", None))
        self.nms_threshold_label.setText(QCoreApplication.translate("DeepFakeMainWindow", u"0.40", None))
        self.detect_btn.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Start Detection", None))
        self.detection_preview.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Detection Preview", None))
        self.prev_btn_2.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Previous", None))
        self.next_btn_2.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Next", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.detection_tab), QCoreApplication.translate("DeepFakeMainWindow", u"2. Face Detection", None))
        self.groupBox_4.setTitle(QCoreApplication.translate("DeepFakeMainWindow", u"Landmark Settings", None))
        self.label_5.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Landmark Model:", None))
        self.landmark_model_combo.setItemText(0, QCoreApplication.translate("DeepFakeMainWindow", u"Model 1", None))
        self.landmark_model_combo.setItemText(1, QCoreApplication.translate("DeepFakeMainWindow", u"Model 2", None))
        self.landmark_model_combo.setItemText(2, QCoreApplication.translate("DeepFakeMainWindow", u"Model 3", None))

        self.label_6.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Landmark Points:", None))
        self.landmark_points_combo.setItemText(0, QCoreApplication.translate("DeepFakeMainWindow", u"68 Points", None))
        self.landmark_points_combo.setItemText(1, QCoreApplication.translate("DeepFakeMainWindow", u"81 Points", None))
        self.landmark_points_combo.setItemText(2, QCoreApplication.translate("DeepFakeMainWindow", u"194 Points", None))

        self.detect_landmarks_btn.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Detect Landmarks", None))
        self.landmarks_preview.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Landmarks Preview", None))
        self.prev_btn_3.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Previous: Detection", None))
        self.next_btn_3.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Next: Alignment", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.landmarks_tab), QCoreApplication.translate("DeepFakeMainWindow", u"3. Landmarks", None))
        self.groupBox_5.setTitle(QCoreApplication.translate("DeepFakeMainWindow", u"Alignment Settings", None))
        self.label_7.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Alignment Method:", None))
        self.alignment_method_combo.setItemText(0, QCoreApplication.translate("DeepFakeMainWindow", u"Similarity Transform", None))
        self.alignment_method_combo.setItemText(1, QCoreApplication.translate("DeepFakeMainWindow", u"Affine Transform", None))
        self.alignment_method_combo.setItemText(2, QCoreApplication.translate("DeepFakeMainWindow", u"Perspective Transform", None))

        self.label_8.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Face Size:", None))
        self.face_size_label.setText(QCoreApplication.translate("DeepFakeMainWindow", u"256", None))
        self.align_faces_btn.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Align Faces", None))
        self.alignment_preview.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Alignment Preview", None))
        self.prev_btn_4.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Previous: Landmarks", None))
        self.next_btn_4.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Next: Recognition", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.alignment_tab), QCoreApplication.translate("DeepFakeMainWindow", u"4. Alignment", None))
        self.groupBox_6.setTitle(QCoreApplication.translate("DeepFakeMainWindow", u"Recognition Settings", None))
        self.label_9.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Recognition Model:", None))
        self.recognition_model_combo.setItemText(0, QCoreApplication.translate("DeepFakeMainWindow", u"ArcFace", None))
        self.recognition_model_combo.setItemText(1, QCoreApplication.translate("DeepFakeMainWindow", u"CosFace", None))
        self.recognition_model_combo.setItemText(2, QCoreApplication.translate("DeepFakeMainWindow", u"FaceNet", None))

        self.label_10.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Similarity Threshold:", None))
        self.similarity_threshold_label.setText(QCoreApplication.translate("DeepFakeMainWindow", u"0.80", None))
        self.recognize_faces_btn.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Recognize Faces", None))
        self.recognition_preview.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Recognition Preview", None))
        self.prev_btn_5.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Previous: Alignment", None))
        self.next_btn_5.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Next: Segmentation", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.recognition_tab), QCoreApplication.translate("DeepFakeMainWindow", u"5. Recognition", None))
        self.groupBox_7.setTitle(QCoreApplication.translate("DeepFakeMainWindow", u"Segmentation Settings", None))
        self.label_11.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Segmentation Model:", None))
        self.segmentation_model_combo.setItemText(0, QCoreApplication.translate("DeepFakeMainWindow", u"BiSeNet", None))
        self.segmentation_model_combo.setItemText(1, QCoreApplication.translate("DeepFakeMainWindow", u"U-Net", None))
        self.segmentation_model_combo.setItemText(2, QCoreApplication.translate("DeepFakeMainWindow", u"DeepLab", None))

        self.label_12.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Mask Blur:", None))
        self.mask_blur_label.setText(QCoreApplication.translate("DeepFakeMainWindow", u"5", None))
        self.segment_faces_btn.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Segment Faces", None))
        self.segmentation_preview.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Segmentation Preview", None))
        self.prev_btn_6.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Previous: Recognition", None))
        self.next_btn_6.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Next: Face Swapping", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.segmentation_tab), QCoreApplication.translate("DeepFakeMainWindow", u"6. Segmentation", None))
        self.groupBox_8.setTitle(QCoreApplication.translate("DeepFakeMainWindow", u"Face Swapping Settings", None))
        self.label_13.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Swap Model:", None))
        self.swap_model_combo.setItemText(0, QCoreApplication.translate("DeepFakeMainWindow", u"SimSwap", None))
        self.swap_model_combo.setItemText(1, QCoreApplication.translate("DeepFakeMainWindow", u"FaceShifter", None))
        self.swap_model_combo.setItemText(2, QCoreApplication.translate("DeepFakeMainWindow", u"FSGAN", None))

        self.label_14.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Blend Ratio:", None))
        self.blend_ratio_label.setText(QCoreApplication.translate("DeepFakeMainWindow", u"0.80", None))
        self.color_correction_check.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Color Correction", None))
        self.start_swap_btn.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Start Face Swapping", None))
        self.swapping_preview.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Face Swapping Preview", None))
        self.prev_btn_7.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Previous: Segmentation", None))
        self.next_btn_7.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Next: Export", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.swapping_tab), QCoreApplication.translate("DeepFakeMainWindow", u"7. Face Swapping", None))
        self.groupBox_9.setTitle(QCoreApplication.translate("DeepFakeMainWindow", u"Export Settings", None))
        self.label_16.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Output Format:", None))
        self.output_format_combo.setItemText(0, QCoreApplication.translate("DeepFakeMainWindow", u"MP4", None))
        self.output_format_combo.setItemText(1, QCoreApplication.translate("DeepFakeMainWindow", u"AVI", None))
        self.output_format_combo.setItemText(2, QCoreApplication.translate("DeepFakeMainWindow", u"MOV", None))
        self.output_format_combo.setItemText(3, QCoreApplication.translate("DeepFakeMainWindow", u"WebM", None))

        self.label_17.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Quality:", None))
        self.quality_label.setText(QCoreApplication.translate("DeepFakeMainWindow", u"8", None))
        self.label_18.setText(QCoreApplication.translate("DeepFakeMainWindow", u"FPS:", None))
        self.fps_combo.setItemText(0, QCoreApplication.translate("DeepFakeMainWindow", u"24", None))
        self.fps_combo.setItemText(1, QCoreApplication.translate("DeepFakeMainWindow", u"25", None))
        self.fps_combo.setItemText(2, QCoreApplication.translate("DeepFakeMainWindow", u"30", None))
        self.fps_combo.setItemText(3, QCoreApplication.translate("DeepFakeMainWindow", u"60", None))

        self.label_19.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Output Path:", None))
        self.output_path_label.setText(QCoreApplication.translate("DeepFakeMainWindow", u"No output path selected", None))
        self.select_output_btn.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Select Output Path", None))
        self.export_btn.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Export Video", None))
        self.export_preview.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Export Preview", None))
        self.prev_btn_8.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Previous: Face Swapping", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.export_tab), QCoreApplication.translate("DeepFakeMainWindow", u"8. Export", None))
        self.play_pause_btn.setText(QCoreApplication.translate("DeepFakeMainWindow", u"\u25b6", None))
        self.stop_btn.setText(QCoreApplication.translate("DeepFakeMainWindow", u"\u23f9", None))
        self.time_label.setText(QCoreApplication.translate("DeepFakeMainWindow", u"00:00 / 00:00", None))
        self.frame_label.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Frame: 0 / 0", None))
        self.label_20.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Timeline:", None))
        self.label_21.setText(QCoreApplication.translate("DeepFakeMainWindow", u"Zoom:", None))
    # retranslateUi

