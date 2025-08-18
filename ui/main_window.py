from .handler import CollapsibleSplitter
from .handler import VideoPreviewWidget, DetectedFacesWidget

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox, QDialog,
    QVBoxLayout, QHBoxLayout, QLabel, QRadioButton, QGroupBox,
    QPushButton, QProgressBar, QTextEdit, QWidget, QSlider,
    QSpinBox, QCheckBox, QComboBox, QStackedWidget, QTabWidget,
    QSizePolicy, QFormLayout, QFrame, QGridLayout, QSpacerItem,
    QSplitter, QSizeGrip, QMenuBar, QMenu, QScrollArea, QStyleOptionButton, QListWidget
)
from PySide6.QtCore import (
    QSize, QEvent, Qt, QThread, Signal, QTimer, 
    QPropertyAnimation, QEasingCurve, QUrl, QMutex, QMutexLocker,
    QCoreApplication, QDate, QDateTime, QLocale, QMetaObject, 
    QObject, QPoint, QRect, QTime
)
from PySide6.QtGui import (
    QPixmap, QImage, QColor, QIcon, QAction, QBrush, 
    QConicalGradient, QCursor, QFont, QFontDatabase, QGradient,
    QKeySequence, QLinearGradient, QPalette, QRadialGradient, QTransform
)

from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget

class Ui_MainWindow(object):
        
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1400, 900)
        MainWindow.setMinimumSize(QSize(1400, 900))
        MainWindow.setStyleSheet(u"""""")
        
        self.actionsave = QAction(MainWindow)
        self.actionsave.setObjectName(u"actionsave")
        self.actionimport_image = QAction(MainWindow)
        self.actionimport_image.setObjectName(u"actionimport_image")
        self.actionimport_video = QAction(MainWindow)
        self.actionimport_video.setObjectName(u"actionimport_video")
        self.actiondelete_source = QAction(MainWindow)
        self.actiondelete_source.setObjectName(u"actiondelete_source")
        self.actiondelete_target = QAction(MainWindow)
        self.actiondelete_target.setObjectName(u"actiondelete_target")
        self.actionprefrances = QAction(MainWindow)
        self.actionprefrances.setObjectName(u"actionprefrances")
        self.actiondoc = QAction(MainWindow)
        self.actiondoc.setObjectName(u"actiondoc")
        self.actionlinks = QAction(MainWindow)
        self.actionlinks.setObjectName(u"actionlinks")
        
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralwidget.setMinimumSize(QSize(1400, 800))
        self.gridLayout_2 = QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName(u"gridLayout_2")

        self.main_splitter = CollapsibleSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setObjectName(u"main_splitter")
        
        left_container = QWidget()
        left_container.setObjectName(u"left_container")
        left_layout = QVBoxLayout(left_container)
        left_layout.setObjectName(u"left_layout")
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        self.tabWidget_2 = QTabWidget(left_container)
        self.tabWidget_2.setObjectName(u"tabWidget_2")
        self.tabWidget_2.setMinimumSize(QSize(250, 600))
        self.tabWidget_2.setMaximumSize(QSize(400, 16777215))
        self.tabWidget_2.setTabPosition(QTabWidget.TabPosition.West)
        self.tabWidget_2.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        self.tabFaceSwapping_3 = QWidget()
        self.tabFaceSwapping_3.setObjectName(u"tabFaceSwapping_3")
        self.faceSwappingLayout_3 = QVBoxLayout(self.tabFaceSwapping_3)
        self.faceSwappingLayout_3.setObjectName(u"faceSwappingLayout_3")
        
        self.label_3 = QLabel(self.tabFaceSwapping_3)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.faceSwappingLayout_3.addWidget(self.label_3)
        
        self.comboBox_3 = QComboBox(self.tabFaceSwapping_3)
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.setObjectName(u"comboBox_3")
        self.faceSwappingLayout_3.addWidget(self.comboBox_3)
        
        self.detectButton_3 = QPushButton(self.tabFaceSwapping_3)
        self.detectButton_3.setObjectName(u"detectButton_3")
        self.faceSwappingLayout_3.addWidget(self.detectButton_3)
        
        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.faceSwappingLayout_3.addItem(self.verticalSpacer)
        
        self.tabWidget_2.addTab(self.tabFaceSwapping_3, "")
        
        self.tabAlignment_3 = QWidget()
        self.tabAlignment_3.setObjectName(u"tabAlignment_3")
        self.alignmentLayout_3 = QVBoxLayout(self.tabAlignment_3)
        self.alignmentLayout_3.setObjectName(u"alignmentLayout_3")
        
        self.alignmentEnableSwitch = QCheckBox("Enable Alignment Settings")
        self.alignmentEnableSwitch.setObjectName(u"alignmentEnableSwitch")
        self.alignmentEnableSwitch.setChecked(False)
        self.alignmentEnableSwitch.setStyleSheet("""
            QCheckBox {
                font-weight: bold;
                padding: 5px;
            }
        """)
        self.alignmentLayout_3.addWidget(self.alignmentEnableSwitch)
        
        self.alignmentSettingsContainer = QWidget()
        self.alignmentSettingsContainer.setObjectName(u"alignmentSettingsContainer")
        self.alignmentSettingsContainer.setEnabled(False)
        alignmentContainerLayout = QVBoxLayout(self.alignmentSettingsContainer)
        
        self.alignTitle_3 = QLabel(self.alignmentSettingsContainer)
        self.alignTitle_3.setObjectName(u"alignTitle_3")
        alignmentContainerLayout.addWidget(self.alignTitle_3)
        
        self.optionsGroup_3 = QGroupBox(self.alignmentSettingsContainer)
        self.optionsGroup_3.setObjectName(u"optionsGroup_3")
        self.optionsLayout_3 = QVBoxLayout(self.optionsGroup_3)
        self.optionsLayout_3.setObjectName(u"optionsLayout_3")
        
        self.precisionLayout_3 = QHBoxLayout()
        self.precisionLayout_3.setObjectName(u"precisionLayout_3")
        self.precisionLabel_3 = QLabel(self.optionsGroup_3)
        self.precisionLabel_3.setObjectName(u"precisionLabel_3")
        self.precisionLayout_3.addWidget(self.precisionLabel_3)
        
        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.precisionLayout_3.addItem(self.horizontalSpacer_5)
        
        self.precisionValue_3 = QLabel(self.optionsGroup_3)
        self.precisionValue_3.setObjectName(u"precisionValue_3")
        self.precisionLayout_3.addWidget(self.precisionValue_3)
        self.optionsLayout_3.addLayout(self.precisionLayout_3)
        
        self.precisionSlider_3 = QSlider(self.optionsGroup_3)
        self.precisionSlider_3.setObjectName(u"precisionSlider_3")
        self.precisionSlider_3.setMinimum(0)
        self.precisionSlider_3.setMaximum(100)
        self.precisionSlider_3.setValue(85)
        self.precisionSlider_3.setOrientation(Qt.Orientation.Horizontal)
        self.optionsLayout_3.addWidget(self.precisionSlider_3)
        
        self.Auto01 = QRadioButton(self.optionsGroup_3)
        self.Auto01.setObjectName(u"Auto01")
        self.Auto01.setChecked(True)
        self.optionsLayout_3.addWidget(self.Auto01)
        
        self.manual_01 = QRadioButton(self.optionsGroup_3)
        self.manual_01.setObjectName(u"manual_01")
        self.optionsLayout_3.addWidget(self.manual_01)
        
        self.Mask01 = QRadioButton(self.optionsGroup_3)
        self.Mask01.setObjectName(u"Mask01")
        self.optionsLayout_3.addWidget(self.Mask01)
        
        self.alignButton_3 = QPushButton(self.optionsGroup_3)
        self.alignButton_3.setObjectName(u"alignButton_3")
        self.optionsLayout_3.addWidget(self.alignButton_3)
        
        alignmentContainerLayout.addWidget(self.optionsGroup_3)
        
        self.alignmentLayout_3.addWidget(self.alignmentSettingsContainer)
        
        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.alignmentLayout_3.addItem(self.verticalSpacer_3)
        
        self.tabWidget_2.addTab(self.tabAlignment_3, "")
        
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.verticalLayout_20 = QVBoxLayout(self.tab_2)
        self.verticalLayout_20.setObjectName(u"verticalLayout_20")
        self.verticalLayout_20.setContentsMargins(10, 10, 10, 10)
        
        self.label_4 = QLabel(self.tab_2)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.verticalLayout_20.addWidget(self.label_4)
        
        self.comboBox_4 = QComboBox(self.tab_2)
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.setObjectName(u"comboBox_4")
        self.verticalLayout_20.addWidget(self.comboBox_4)
        
        self.blendLayout_3 = QHBoxLayout()
        self.blendLayout_3.setObjectName(u"blendLayout_3")
        self.blendLabel_3 = QLabel(self.tab_2)
        self.blendLabel_3.setObjectName(u"blendLabel_3")
        self.blendLayout_3.addWidget(self.blendLabel_3)
        
        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.blendLayout_3.addItem(self.horizontalSpacer_4)
        
        self.blendValue_3 = QLabel(self.tab_2)
        self.blendValue_3.setObjectName(u"blendValue_3")
        self.blendLayout_3.addWidget(self.blendValue_3)
        self.verticalLayout_20.addLayout(self.blendLayout_3)
        
        self.blendSlider_3 = QSlider(self.tab_2)
        self.blendSlider_3.setObjectName(u"blendSlider_3")
        self.blendSlider_3.setMinimum(0)
        self.blendSlider_3.setMaximum(100)
        self.blendSlider_3.setValue(84)
        self.blendSlider_3.setOrientation(Qt.Orientation.Horizontal)
        self.verticalLayout_20.addWidget(self.blendSlider_3)
        
        self.edgeLayout_3 = QHBoxLayout()
        self.edgeLayout_3.setObjectName(u"edgeLayout_3")
        self.edgeLabel_3 = QLabel(self.tab_2)
        self.edgeLabel_3.setObjectName(u"edgeLabel_3")
        self.edgeLayout_3.addWidget(self.edgeLabel_3)
        
        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.edgeLayout_3.addItem(self.horizontalSpacer_3)
        
        self.edgeValue_3 = QLabel(self.tab_2)
        self.edgeValue_3.setObjectName(u"edgeValue_3")
        self.edgeLayout_3.addWidget(self.edgeValue_3)
        self.verticalLayout_20.addLayout(self.edgeLayout_3)
        
        self.edgeSlider_3 = QSlider(self.tab_2)
        self.edgeSlider_3.setObjectName(u"edgeSlider_3")
        self.edgeSlider_3.setMinimum(0)
        self.edgeSlider_3.setMaximum(100)
        self.edgeSlider_3.setValue(48)
        self.edgeSlider_3.setOrientation(Qt.Orientation.Horizontal)
        self.verticalLayout_20.addWidget(self.edgeSlider_3)
        
        self.swapButton_3 = QPushButton(self.tab_2)
        self.swapButton_3.setObjectName(u"swapButton_3")
        self.verticalLayout_20.addWidget(self.swapButton_3)
        
        self.verticalSpacer_8 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.verticalLayout_20.addItem(self.verticalSpacer_8)
        
        self.tabWidget_2.addTab(self.tab_2, "")
        
        self.tabFaceEnhancement_3 = QWidget()
        self.tabFaceEnhancement_3.setObjectName(u"tabFaceEnhancement_3")
        self.verticalLayout_25 = QVBoxLayout(self.tabFaceEnhancement_3)
        self.verticalLayout_25.setObjectName(u"verticalLayout_25")
        self.verticalLayout_25.setContentsMargins(10, 10, 10, 10)
        
        self.enhancementEnableSwitch = QCheckBox("Enable Enhancement Settings")
        self.enhancementEnableSwitch.setObjectName(u"enhancementEnableSwitch")
        self.enhancementEnableSwitch.setChecked(False)
        self.enhancementEnableSwitch.setStyleSheet("""
            QCheckBox {
                font-weight: bold;
                padding: 5px;
            }
        """)
        self.verticalLayout_25.addWidget(self.enhancementEnableSwitch)
        
        self.enhancementSettingsContainer = QWidget()
        self.enhancementSettingsContainer.setObjectName(u"enhancementSettingsContainer")
        self.enhancementSettingsContainer.setEnabled(False)
        enhancementContainerLayout = QVBoxLayout(self.enhancementSettingsContainer)
        
        self.horizontalSpacer_6 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        
        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        
        self.horizontalSpacer_7 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.horizontalLayout_8.addItem(self.horizontalSpacer_7)
        
        enhancementContainerLayout.addLayout(self.horizontalLayout_8)
        
        self.applyEnhanceButton_3 = QPushButton(self.enhancementSettingsContainer)
        self.applyEnhanceButton_3.setObjectName(u"applyEnhanceButton_3")
        
        self.adjustmentsButton_01 = QPushButton(self.enhancementSettingsContainer)
        self.adjustmentsButton_01.setObjectName(u"adjustmentsButton_01")
        enhancementContainerLayout.addWidget(self.adjustmentsButton_01)
        
        enhancementContainerLayout.addWidget(self.applyEnhanceButton_3)
        
        self.verticalLayout_25.addWidget(self.enhancementSettingsContainer)
        
        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.verticalLayout_25.addItem(self.verticalSpacer_2)
        
        self.tabWidget_2.addTab(self.tabFaceEnhancement_3, "")
        
        # self.tabLandmarksEditor_01 = QWidget()
        # self.tabLandmarksEditor_01.setObjectName(u"tabLandmarksEditor_01")
        # self.gridLayout = QGridLayout(self.tabLandmarksEditor_01)
        # self.gridLayout.setObjectName(u"gridLayout")
        
        # self.LandmarksEditor_01 = QLabel(self.tabLandmarksEditor_01)
        # self.LandmarksEditor_01.setObjectName(u"LandmarksEditor_01")
        # self.LandmarksEditor_01.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # self.gridLayout.addWidget(self.LandmarksEditor_01, 0, 0, 1, 1)
        
        # self.horizontalSpacer_8 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        
        # self.btnLandmarksEditor = QPushButton(self.tabLandmarksEditor_01)
        # self.btnLandmarksEditor.setObjectName(u"btnLandmarksEditor")
        # self.gridLayout.addWidget(self.btnLandmarksEditor, 6, 0, 1, 1)
        
        # self.verticalSpacer_4 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        # self.gridLayout.addItem(self.verticalSpacer_4, 8, 0, 1, 1)
        
        # self.tabWidget_2.addTab(self.tabLandmarksEditor_01, "")
        
        # Face Aging Tab
        self.FaceAging_tab = QWidget()
        self.FaceAging_tab.setObjectName(u"FaceAging_tab")
        self.verticalLayout3D = QVBoxLayout(self.FaceAging_tab)
        self.verticalLayout3D.setObjectName(u"verticalLayout3D")
        self.verticalLayout3D.setContentsMargins(10, 10, 10, 10)
        
        self.FaceAgingEnableSwitch = QCheckBox("Enable Face Ageing")
        self.FaceAgingEnableSwitch.setObjectName(u"FaceAgingEnableSwitch")
        self.FaceAgingEnableSwitch.setChecked(False)
        self.FaceAgingEnableSwitch.setStyleSheet("""
            QCheckBox {
                font-weight: bold;
                padding: 5px;
            }
        """)
        self.verticalLayout3D.addWidget(self.FaceAgingEnableSwitch)
        
        self.Face_AgeSettingsContainer = QWidget()
        self.Face_AgeSettingsContainer.setObjectName(u"Face_AgeSettingsContainer")
        self.Face_AgeSettingsContainer.setEnabled(False)
        alignment3DContainerLayout = QVBoxLayout(self.Face_AgeSettingsContainer)
        
        self.btnAgeEditor = QPushButton("Face Ageing") 
        self.btnAgeEditor.setObjectName(u"btnAgeEditor")
        self.btnAgeEditor.setStyleSheet("""  """)
        
        alignment3DContainerLayout.addWidget(self.btnAgeEditor)
        
        self.verticalLayout3D.addWidget(self.Face_AgeSettingsContainer)
        
        self.verticalSpacer3D = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.verticalLayout3D.addItem(self.verticalSpacer3D)
        
        self.tabWidget_2.addTab(self.FaceAging_tab, "")
        
        # Sound Tab
        self.sound_tab = QWidget()
        self.sound_tab.setObjectName(u"sound_tab")
        self.verticalLayout_Sound = QVBoxLayout(self.sound_tab)
        self.verticalLayout_Sound.setObjectName(u"verticalLayout_Sound")
        self.verticalLayout_Sound.setContentsMargins(10, 10, 10, 10)
        
        self.soundswitch = QCheckBox("Enable Sound Settings")
        self.soundswitch.setObjectName(u"soundswitch")
        self.soundswitch.setChecked(False)
        self.soundswitch.setStyleSheet("""
            QCheckBox {
                font-weight: bold;
                padding: 5px;
            }
        """)
        self.verticalLayout_Sound.addWidget(self.soundswitch)
        
        self.soundSettingsContainer = QWidget()
        self.soundSettingsContainer.setObjectName(u"soundSettingsContainer")
        self.soundSettingsContainer.setEnabled(False)
        soundContainerLayout = QVBoxLayout(self.soundSettingsContainer)
        
        # Sound processing button
        self.btnSoundProcessor = QPushButton("Process Audio")
        self.btnSoundProcessor.setObjectName(u"btnSoundProcessor")
        self.btnSoundProcessor.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        soundContainerLayout.addWidget(self.btnSoundProcessor)
        
        # Volume control
        volumeLayout = QHBoxLayout()
        volumeLabel = QLabel("Volume:")
        volumeLabel.setObjectName(u"volumeLabel")
        self.volumeSlider = QSlider(Qt.Orientation.Horizontal)
        self.volumeSlider.setObjectName(u"volumeSlider")
        self.volumeSlider.setMinimum(0)
        self.volumeSlider.setMaximum(100)
        self.volumeSlider.setValue(50)
        self.volumeValue = QLabel("50")
        self.volumeValue.setObjectName(u"volumeValue")
        
        volumeLayout.addWidget(volumeLabel)
        volumeLayout.addWidget(self.volumeSlider)
        volumeLayout.addWidget(self.volumeValue)
        soundContainerLayout.addLayout(volumeLayout)
        
        self.verticalLayout_Sound.addWidget(self.soundSettingsContainer)
        
        self.verticalSpacerSound = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.verticalLayout_Sound.addItem(self.verticalSpacerSound)
        
        self.tabWidget_2.addTab(self.sound_tab, "")
        
        # 3D Mesh Tab
        self.mesh_3d_tab = QWidget()
        self.mesh_3d_tab.setObjectName(u"mesh_3d_tab")
        self.verticalLayout3DMesh = QVBoxLayout(self.mesh_3d_tab)
        self.verticalLayout3DMesh.setObjectName(u"verticalLayout3DMesh")
        self.verticalLayout3DMesh.setContentsMargins(10, 10, 10, 10)
        
        self.mesh3DEnableSwitch = QCheckBox("Enable 3D Mesh Settings")
        self.mesh3DEnableSwitch.setObjectName(u"mesh3DEnableSwitch")
        self.mesh3DEnableSwitch.setChecked(False)
        self.mesh3DEnableSwitch.setStyleSheet("""
            QCheckBox {
                font-weight: bold;
                padding: 5px;
            }
        """)
        self.verticalLayout3DMesh.addWidget(self.mesh3DEnableSwitch)
        
        self.mesh3DSettingsContainer = QWidget()
        self.mesh3DSettingsContainer.setObjectName(u"mesh3DSettingsContainer")
        self.mesh3DSettingsContainer.setEnabled(False)
        mesh3DContainerLayout = QVBoxLayout(self.mesh3DSettingsContainer)
        
        # Mesh generator button
        self.btnMeshGenerator = QPushButton("Generate 3D Mesh")
        self.btnMeshGenerator.setObjectName(u"btnMeshGenerator")
        self.btnMeshGenerator.setStyleSheet("""
            QPushButton {
                background-color: #5D3FD3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4B0082;
            }
        """)
        mesh3DContainerLayout.addWidget(self.btnMeshGenerator)
        
        self.verticalLayout3DMesh.addWidget(self.mesh3DSettingsContainer)
        
        self.verticalSpacer3DMesh = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.verticalLayout3DMesh.addItem(self.verticalSpacer3DMesh)
        
        self.tabWidget_2.addTab(self.mesh_3d_tab, "")
        
        left_layout.addWidget(self.tabWidget_2)
        left_container.setMinimumWidth(250)
        left_container.setMaximumWidth(400)
        self.main_splitter.addCollapsibleWidget(left_container)
        
        center_container = QWidget()
        center_container.setObjectName(u"center_container")
        center_layout = QVBoxLayout(center_container)
        center_layout.setObjectName(u"center_layout")
        center_layout.setContentsMargins(5, 5, 5, 5)
        
        self.video_preview_widget = VideoPreviewWidget(center_container)
        self.video_preview_widget.setObjectName(u"video_preview_widget")
        center_layout.addWidget(self.video_preview_widget, 1)
        
        self.detected_faces_widget = DetectedFacesWidget(center_container)
        self.detected_faces_widget.setObjectName(u"detected_faces_widget")
        self.detected_faces_widget.setMinimumHeight(150)
        self.detected_faces_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        center_layout.addWidget(self.detected_faces_widget)
        
        self.preview_controls = QGroupBox("Preview Source")
        preview_layout = QHBoxLayout(self.preview_controls)

        self.target_preview_radio = QRadioButton("Target")
        self.swapped_preview_radio = QRadioButton("Swapped")
        self.target_preview_radio.setChecked(True)

        preview_layout.addWidget(self.target_preview_radio)
        preview_layout.addWidget(self.swapped_preview_radio)

        center_layout.addWidget(self.preview_controls)
        center_container.setMinimumWidth(800)
        center_container.setMinimumHeight(600)
        center_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.main_splitter.addCollapsibleWidget(center_container)
        
        right_container = QWidget()
        right_container.setObjectName(u"right_container")
        right_layout = QVBoxLayout(right_container)
        right_layout.setObjectName(u"right_layout")
        right_layout.setContentsMargins(5, 5, 5, 5)
        
        self.tabWidget = QTabWidget(right_container)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setMinimumSize(QSize(250, 600))
        self.tabWidget.setMaximumSize(QSize(400, 16777215))
        self.tabWidget.setTabPosition(QTabWidget.TabPosition.East)
        self.tabWidget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.formLayout = QFormLayout(self.tab)
        self.formLayout.setObjectName(u"formLayout")
        
        self.verticalLayout_11 = QVBoxLayout()
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        
        self.sourceGroup_2 = QGroupBox(self.tab)
        self.sourceGroup_2.setObjectName(u"sourceGroup_2")
        self.sourceLayout_4 = QVBoxLayout(self.sourceGroup_2)
        self.sourceLayout_4.setObjectName(u"sourceLayout_4")
        
        self.dropSource_4 = QLabel(self.sourceGroup_2)
        self.dropSource_4.setObjectName(u"dropSource_4")
        self.dropSource_4.setMaximumSize(QSize(300, 300))
        self.dropSource_4.setStyleSheet("border: 3px dashed #FFFFFF; border-radius: 16px; font-size: 16px; color: #FFFFFF;")
        self.dropSource_4.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dropSource_4.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.dropSource_4.setWordWrap(True)
        self.sourceLayout_4.addWidget(self.dropSource_4)
        
        self.verticalLayout_11.addWidget(self.sourceGroup_2)
        
        self.targetGroup_2 = QGroupBox(self.tab)
        self.targetGroup_2.setObjectName(u"targetGroup_2")
        self.targetLayout_2 = QVBoxLayout(self.targetGroup_2)
        self.targetLayout_2.setObjectName(u"targetLayout_2")
        
        self.dropTarget_2 = QLabel(self.targetGroup_2)
        self.dropTarget_2.setObjectName(u"dropTarget_2")
        self.dropTarget_2.setMaximumSize(QSize(300, 300))
        self.dropTarget_2.setStyleSheet("border: 3px dashed #FFFFFF; border-radius: 16px; font-size: 16px; color: #FFFFFF;")
        self.dropTarget_2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dropTarget_2.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.dropTarget_2.setWordWrap(True)
        self.targetLayout_2.addWidget(self.dropTarget_2)
        
        self.verticalLayout_11.addWidget(self.targetGroup_2)
        
        self.formLayout.setLayout(0, QFormLayout.ItemRole.SpanningRole, self.verticalLayout_11)
        
        self.tabWidget.addTab(self.tab, "")
        
        progress_tab = QWidget()
        progress_tab.setObjectName(u"progress_tab")
        progress_layout = QVBoxLayout(progress_tab)
        progress_layout.setContentsMargins(10, 10, 10, 10)
        progress_layout.setSpacing(8)
        
        overall_label = QLabel("Overall Progress:")
        overall_label.setStyleSheet("font-weight: bold; color: #0d7ae7;")
        self.overall_progress = QProgressBar()
        self.overall_progress.setRange(0, 100)
        self.overall_progress.setTextVisible(True)
        
        progress_layout.addWidget(overall_label)
        progress_layout.addWidget(self.overall_progress)
        
        task_label = QLabel("Current Task:")
        task_label.setStyleSheet("font-weight: bold; color: #0d7ae7;")
        self.task_progress = QProgressBar()
        self.task_progress.setRange(0, 100)
        self.task_progress.setTextVisible(True)
        
        progress_layout.addWidget(task_label)
        progress_layout.addWidget(self.task_progress)
        
        resources_group = QGroupBox("System Resources")
        resources_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        resources_layout = QVBoxLayout(resources_group)
        resources_layout.setSpacing(5)
        
        cpu_layout = QHBoxLayout()
        cpu_label = QLabel("CPU:")
        cpu_label.setMinimumWidth(50)
        self.cpu_progress = QProgressBar()
        self.cpu_progress.setRange(0, 100)
        self.cpu_progress.setTextVisible(True)
        self.cpu_progress.setStyleSheet("QProgressBar::chunk { background-color: #00c851; }")
        cpu_layout.addWidget(cpu_label)
        cpu_layout.addWidget(self.cpu_progress)
        resources_layout.addLayout(cpu_layout)
        
        ram_layout = QHBoxLayout()
        ram_label = QLabel("RAM:")
        ram_label.setMinimumWidth(50)
        self.ram_progress = QProgressBar()
        self.ram_progress.setRange(0, 100)
        self.ram_progress.setTextVisible(True)
        self.ram_progress.setStyleSheet("QProgressBar::chunk { background-color: #ff4444; }")
        ram_layout.addWidget(ram_label)
        ram_layout.addWidget(self.ram_progress)
        resources_layout.addLayout(ram_layout)
        
        gpu_layout = QHBoxLayout()
        gpu_label = QLabel("GPU:")
        gpu_label.setMinimumWidth(50)
        self.gpu_progress = QProgressBar()
        self.gpu_progress.setRange(0, 100)
        self.gpu_progress.setTextVisible(True)
        self.gpu_progress.setStyleSheet("QProgressBar::chunk { background-color: #ffbb33; }")
        gpu_layout.addWidget(gpu_label)
        gpu_layout.addWidget(self.gpu_progress)
        resources_layout.addLayout(gpu_layout)
        
        vram_layout = QHBoxLayout()
        vram_label = QLabel("VRAM:")
        vram_label.setMinimumWidth(50)
        self.vram_progress = QProgressBar()
        self.vram_progress.setRange(0, 100)
        self.vram_progress.setTextVisible(True)
        self.vram_progress.setStyleSheet("QProgressBar::chunk { background-color: #aa66cc; }")
        vram_layout.addWidget(vram_label)
        vram_layout.addWidget(self.vram_progress)
        resources_layout.addLayout(vram_layout)
        
        progress_layout.addWidget(resources_group)
        
        log_label = QLabel("Processing Log:")
        log_label.setStyleSheet("font-weight: bold; color: #0d7ae7;")
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(600)
        
        progress_layout.addWidget(log_label)
        progress_layout.addWidget(self.log_output)
        progress_layout.addStretch()
        
        self.tabWidget.addTab(progress_tab, "Progress")
        
        self.widget = QWidget()
        self.widget.setObjectName(u"widget")
        self.verticalLayout_14 = QVBoxLayout(self.widget)
        self.verticalLayout_14.setObjectName(u"verticalLayout_14")
        self.verticalLayout_14.setContentsMargins(10, 10, 10, 10)
        
        self.label_2 = QLabel(self.widget)
        self.label_2.setObjectName(u"label_2")
        self.verticalLayout_14.addWidget(self.label_2)
        
        self.comboBox_2 = QComboBox(self.widget)
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.setObjectName(u"comboBox_2")
        self.verticalLayout_14.addWidget(self.comboBox_2)
        
        self.outputLabel_2 = QLabel(self.widget)
        self.outputLabel_2.setObjectName(u"outputLabel_2")
        self.verticalLayout_14.addWidget(self.outputLabel_2)
        
        self.outputPath_2 = QLabel(self.widget)
        self.outputPath_2.setObjectName(u"outputPath_2")
        self.verticalLayout_14.addWidget(self.outputPath_2)
        
        self.columnsLabel_2 = QLabel(self.widget)
        self.columnsLabel_2.setObjectName(u"columnsLabel_2")
        self.verticalLayout_14.addWidget(self.columnsLabel_2)
        
        self.columnsCombo_2 = QComboBox(self.widget)
        self.columnsCombo_2.addItem("")
        self.columnsCombo_2.addItem("")
        self.columnsCombo_2.addItem("")
        self.columnsCombo_2.addItem("")
        self.columnsCombo_2.setObjectName(u"columnsCombo_2")
        self.verticalLayout_14.addWidget(self.columnsCombo_2)
        
        self.exportButton_2 = QPushButton(self.widget)
        self.exportButton_2.setObjectName(u"exportButton_2")
        self.verticalLayout_14.addWidget(self.exportButton_2)
        
        self.verticalSpacer_5 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.verticalLayout_14.addItem(self.verticalSpacer_5)
        
        self.tabWidget.addTab(self.widget, "")
        
        right_layout.addWidget(self.tabWidget)
        right_container.setMinimumWidth(250)
        right_container.setMaximumWidth(400)
        self.main_splitter.addCollapsibleWidget(right_container)
        
        self.main_splitter.setSizes([280, 1200, 280])
        
        left_container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        center_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        
        self.gridLayout_2.addWidget(self.main_splitter, 0, 0, 1, 1)
        
        MainWindow.setCentralWidget(self.centralwidget)
        
        self.menuBar_2 = QMenuBar(MainWindow)
        self.menuBar_2.setObjectName(u"menuBar_2")
        self.menuBar_2.setGeometry(QRect(0, 0, 1400, 33))
        self.menufile = QMenu(self.menuBar_2)
        self.menufile.setObjectName(u"menufile")
        self.menuedit = QMenu(self.menuBar_2)
        self.menuedit.setObjectName(u"menuedit")
        self.menusettings = QMenu(self.menuBar_2)
        self.menusettings.setObjectName(u"menusettings")
        self.menuHelp = QMenu(self.menuBar_2)
        self.menuHelp.setObjectName(u"menuHelp")
        MainWindow.setMenuBar(self.menuBar_2)
        
        self.menuBar_2.addAction(self.menufile.menuAction())
        self.menuBar_2.addAction(self.menuedit.menuAction())
        self.menuBar_2.addAction(self.menusettings.menuAction())
        self.menuBar_2.addAction(self.menuHelp.menuAction())
        self.menufile.addAction(self.actionsave)
        self.menufile.addAction(self.actionimport_image)
        self.menufile.addAction(self.actionimport_video)
        self.menuedit.addAction(self.actiondelete_source)
        self.menuedit.addAction(self.actiondelete_target)
        self.menusettings.addAction(self.actionprefrances)
        self.menuHelp.addAction(self.actiondoc)
        self.menuHelp.addAction(self.actionlinks)

        self.retranslateUi(MainWindow)
        
        # Connect signals
        self.blendSlider_3.valueChanged.connect(self.blendValue_3.setNum)
        self.edgeSlider_3.valueChanged.connect(self.edgeValue_3.setNum)
        self.precisionSlider_3.valueChanged.connect(self.precisionValue_3.setNum)
        self.volumeSlider.valueChanged.connect(self.volumeValue.setNum)
        
        self.alignmentEnableSwitch.toggled.connect(self.alignmentSettingsContainer.setEnabled)
        self.enhancementEnableSwitch.toggled.connect(self.enhancementSettingsContainer.setEnabled)
        self.FaceAgingEnableSwitch.toggled.connect(self.Face_AgeSettingsContainer.setEnabled)
        self.soundswitch.toggled.connect(self.soundSettingsContainer.setEnabled)
        self.mesh3DEnableSwitch.toggled.connect(self.mesh3DSettingsContainer.setEnabled)
        
        self.tabWidget_2.setCurrentIndex(0)
        self.tabWidget.setCurrentIndex(0)

        QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Face Swapping Studio - AI Powered", None))
        MainWindow.setWindowIcon(QIcon("Assets/Icons/FaceSwapingLogo.png"))

        self.actionsave.setText(QCoreApplication.translate("MainWindow", u"Save", None))
        self.actionimport_image.setText(QCoreApplication.translate("MainWindow", u"Import Image", None))
        self.actionimport_video.setText(QCoreApplication.translate("MainWindow", u"Import Video", None))
        self.actiondelete_source.setText(QCoreApplication.translate("MainWindow", u"Delete Source", None))
        self.actiondelete_target.setText(QCoreApplication.translate("MainWindow", u"Delete Target", None))
        self.actionprefrances.setText(QCoreApplication.translate("MainWindow", u"Preferences", None))
        self.actiondoc.setText(QCoreApplication.translate("MainWindow", u"Documentation", None))
        self.actionlinks.setText(QCoreApplication.translate("MainWindow", u"Links", None))
        
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Detection Model", None))
        self.comboBox_3.setItemText(0, QCoreApplication.translate("MainWindow", u"Yolo", None))
        self.comboBox_3.setItemText(1, QCoreApplication.translate("MainWindow", u"SCRFD", None))
        self.comboBox_3.setItemText(2, QCoreApplication.translate("MainWindow", u"Retina Face", None))
        
        self.detectButton_3.setText(QCoreApplication.translate("MainWindow", u"Detect Faces in Target", None))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tabFaceSwapping_3), QCoreApplication.translate("MainWindow", u"Detection", None))
        
        self.alignTitle_3.setText(QCoreApplication.translate("MainWindow", u"Face Alignment", None))
        self.optionsGroup_3.setTitle(QCoreApplication.translate("MainWindow", u"Alignment Options", None))
        self.precisionLabel_3.setText(QCoreApplication.translate("MainWindow", u"Precision", None))
        self.precisionValue_3.setText(QCoreApplication.translate("MainWindow", u"85", None))
        self.Auto01.setText(QCoreApplication.translate("MainWindow", u"Auto", None))
        self.manual_01.setText(QCoreApplication.translate("MainWindow", u"Manual System", None))
        self.Mask01.setText(QCoreApplication.translate("MainWindow", u"Mask", None))
        self.alignButton_3.setText(QCoreApplication.translate("MainWindow", u"Align Faces / Draw Mask", None))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tabAlignment_3), QCoreApplication.translate("MainWindow", u"Alignment", None))
        
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Swap Model", None))
        self.comboBox_4.setItemText(0, QCoreApplication.translate("MainWindow", u"InSwapper_128", None))
        self.comboBox_4.setItemText(1, QCoreApplication.translate("MainWindow", u"InSwapper_256", None))
        self.comboBox_4.setItemText(2, QCoreApplication.translate("MainWindow", u"InSwapper_512", None))
        
        self.blendLabel_3.setText(QCoreApplication.translate("MainWindow", u"Blend Strength", None))
        self.blendValue_3.setText(QCoreApplication.translate("MainWindow", u"84", None))
        self.edgeLabel_3.setText(QCoreApplication.translate("MainWindow", u"Edge Softness", None))
        self.edgeValue_3.setText(QCoreApplication.translate("MainWindow", u"48", None))
        self.swapButton_3.setText(QCoreApplication.translate("MainWindow", u"Perform Face Swap", None))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_2), QCoreApplication.translate("MainWindow", u"Swapping", None))
       
        self.applyEnhanceButton_3.setText(QCoreApplication.translate("MainWindow", u"Enhance Swapped Face", None))
        self.adjustmentsButton_01.setText(QCoreApplication.translate("MainWindow", u"Manual Adjustments", None))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tabFaceEnhancement_3), QCoreApplication.translate("MainWindow", u"Face Enhancement", None))
       
        # self.LandmarksEditor_01.setText(QCoreApplication.translate("MainWindow", u"Landmarks Model", None))
        # self.btnLandmarksEditor.setText(QCoreApplication.translate("MainWindow", u"Landmarks Editor", None))
        # self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tabLandmarksEditor_01), QCoreApplication.translate("MainWindow", u"Landmarks Editor", None))
        
        self.btnAgeEditor.setText(QCoreApplication.translate("MainWindow", u"Face Ageing", None))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.FaceAging_tab), QCoreApplication.translate("MainWindow", u"Face Ageing", None))
        
        # Sound tab translations
        self.btnSoundProcessor.setText(QCoreApplication.translate("MainWindow", u"Process Audio", None))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.sound_tab), QCoreApplication.translate("MainWindow", u"Sound", None))
        
        # 3D Mesh tab translations
        self.btnMeshGenerator.setText(QCoreApplication.translate("MainWindow", u"Generate 3D Mesh", None))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.mesh_3d_tab), QCoreApplication.translate("MainWindow", u"3D Mesh", None))
        
        self.sourceGroup_2.setTitle(QCoreApplication.translate("MainWindow", u"Source Face", None))
        self.dropSource_4.setText(QCoreApplication.translate("MainWindow", u"Drop/Click to select Source Face Image", None))
        self.targetGroup_2.setTitle(QCoreApplication.translate("MainWindow", u"Target Video", None))
        self.dropTarget_2.setText(QCoreApplication.translate("MainWindow", u"Drop/Click to select Target Video", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QCoreApplication.translate("MainWindow", u"Import", None))
        
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Output Format", None))
        self.comboBox_2.setItemText(0, QCoreApplication.translate("MainWindow", u"MP4", None))
        self.comboBox_2.setItemText(1, QCoreApplication.translate("MainWindow", u"MOV", None))
        self.comboBox_2.setItemText(2, QCoreApplication.translate("MainWindow", u"MKV", None))
        
        self.outputLabel_2.setText(QCoreApplication.translate("MainWindow", u"Output Path:", None))
        self.outputPath_2.setText(QCoreApplication.translate("MainWindow", u"Click Export to choose...", None))

        self.columnsLabel_2.setText(QCoreApplication.translate("MainWindow", u"Quality Level:", None))
        self.columnsCombo_2.setItemText(0, QCoreApplication.translate("MainWindow", u"High", None))
        self.columnsCombo_2.setItemText(1, QCoreApplication.translate("MainWindow", u"Medium", None))
        self.columnsCombo_2.setItemText(2, QCoreApplication.translate("MainWindow", u"Low", None))
        self.columnsCombo_2.setItemText(3, QCoreApplication.translate("MainWindow", u"Custom", None))
        
        self.exportButton_2.setText(QCoreApplication.translate("MainWindow", u"Export Final Video", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.widget), QCoreApplication.translate("MainWindow", u"Export", None))
        
        self.menufile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.menuedit.setTitle(QCoreApplication.translate("MainWindow", u"Edit", None))
        self.menusettings.setTitle(QCoreApplication.translate("MainWindow", u"Settings", None))
        self.menuHelp.setTitle(QCoreApplication.translate("MainWindow", u"Help", None))
        