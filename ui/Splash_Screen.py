import sys
import time
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QSplashScreen, QWidget, QLabel, QComboBox, QVBoxLayout, QPushButton, QButtonGroup
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QPixmap, QMovie, QIcon

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class CollapsibleTabBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(50)
        self.setStyleSheet("""
            background-color: #22262b;
            border-right: 1px solid #444;
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 10, 0, 10)
        layout.setSpacing(5)
        self.setLayout(layout)
        
        self.tab_buttons = []
        self.button_group = QButtonGroup()
        
        tabs = [
            ("Swap", "Face Swapping", QIcon()),
            ("Enhance", "Face Enhancement", QIcon()),
            ("Align", "Alignment", QIcon()),
            ("Deep", "Deepfake", QIcon())
        ]
        
        for text, tooltip, icon in tabs:
            btn = QPushButton(text)
            btn.setToolTip(tooltip)
            btn.setFixedSize(40, 40)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #2c3138;
                    border: 1px solid #444;
                    border-radius: 5px;
                    color: #aaa;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #3a4049;
                }
                QPushButton:checked {
                    background-color: #0d4ae7;
                    color: white;
                }
            """)
            btn.setCheckable(True)
            self.tab_buttons.append(btn)
            self.button_group.addButton(btn)
            layout.addWidget(btn)
        
        if self.tab_buttons:
            self.tab_buttons[0].setChecked(True)
        
        layout.addStretch(1)


class FaceSwappingApp(QMainWindow):
    def __init__(self):
        super().__init__()
    
        self.setCentralWidget(self.main_window)
        self.setWindowTitle("Face Swapping App")
        self.setGeometry(100, 100, 1200, 800)

        self.theme_combo = self.findChild(QComboBox, "combo_theme")
        if self.theme_combo:
            self.theme_combo.currentTextChanged.connect(self.change_theme)
        self.change_theme(self.theme_combo.currentText() if self.theme_combo else "Dark")

    def change_theme(self, theme):
        if theme == "Light":
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: #f5f5f5;
                    color: #222;
                    font-family: 'Segoe UI', Arial;
                }
                QPushButton {
                    background-color: #e0e0e0;
                    border: 1px solid #bbb;
                    border-radius: 4px;
                    color: #222;
                }
                QPushButton:hover {
                    background-color: #d0d0d0;
                }
                QLabel {
                    color: #222;
                }
                QComboBox {
                    background-color: #fff;
                    color: #222;
                }
            """)
        else:
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: #101115;
                    color: #ffffff;
                    font-family: 'Segoe UI', Arial;
                }
                QPushButton {
                    background-color: #22262b;
                    border: 1px solid #444;
                    border-radius: 4px;
                    color: #fff;
                }
                QPushButton:hover {
                    background-color: #2c3138;
                }
                QLabel {
                    color: #fff;
                }
                QComboBox {
                    background-color: #22262b;
                    color: #fff;
                }
            """)

class GifSplashScreen(QWidget):
    def __init__(self, gif_path, duration=15000, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(600, 400)
        self.setStyleSheet("background: transparent;")
        self.label = QLabel(self)
        self.label.setGeometry(0, 0, 600, 400)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.movie = QMovie(gif_path)
        self.label.setMovie(self.movie)
        self.movie.start()
        
        
        self.message = QLabel("Face Swapping App\nLoading...", self)
        
        
        self.message.setStyleSheet("color: white; font-size: 10px; font-weight: bold; background: transparent;")
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignBottom)
        self.message.setGeometry(0, 255, 600, 100)

        QTimer.singleShot(duration, self.close)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    gif_path = (r"C:\Users\ROG\Downloads\New folder (4)\Assets\Gif\Face Detection.gif")
    splash = GifSplashScreen(gif_path, duration=15000)
    splash.show()
    # app.processEvents()

    def show_main():
        window = FaceSwappingApp()
        window.show()
        sys.exit(app.exec())

    QTimer.singleShot(15000, show_main)
    app.exec()