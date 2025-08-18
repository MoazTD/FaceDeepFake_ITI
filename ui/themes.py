from uitls import logger

import sys
import random

colors = [
    ["#3498db", "#518142", "#ff0000"],
    ["#04b97f", "#007b50", "#37efba"]
]

current_colors = random.choice(colors)
pos = 0.0
 

 
def update_gradient():
    global pos, current_colors
    pos += 0.01
    if pos > 1.0:
        pos = 0.0
        
def apply_theme(self, theme: str):
        """Apply theme to application"""
        self.current_theme = theme
        
        if theme == "dark":
            stylesheet = """
         QWidget {
    background-color: #121212;
    color: #e0e0e0;
    selection-background-color: #7e57c2;
    selection-color: #ffffff;
    font-family: 'Segoe UI', system-ui;
    font-size: 10pt;
    font-weight: 400;
}

/* Main Window */
QMainWindow {
    background-color: #121212;
    border: 1px solid #2a2a2a;
}

/* Text Inputs - Professional Style */
QLineEdit, QTextEdit, QPlainTextEdit {
    background-color: #1e1e1e;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 8px 12px;
    selection-background-color: #7e57c2;
}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
    border: 1px solid #7e57c2;
    box-shadow: 0 0 0 2px rgba(126, 87, 194, 0.2);
}

/* Enhanced Buttons */
QPushButton {
    background-color: #2a2a2a;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 8px 16px;
    min-width: 90px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #333;
    border: 1px solid #444;
}

QPushButton:pressed {
    background-color: #7e57c2;
    color: white;
    border-color: #7e57c2;
}

QPushButton:flat {
    border: none;
    background: transparent;
}

/* Professional Tabs */
QTabWidget::pane {
    border: none;
    background: #1e1e1e;
    border-radius: 8px;
    margin-top: -1px;
}

QTabBar::tab {
    background: #252525;
    color: #aaa;
    padding: 12px 24px;
    margin-right: 2px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    font-weight: 500;
    border-bottom: 2px solid transparent;
}

QTabBar::tab:selected {
    background: #1e1e1e;
    color: #fff;
    border-bottom: 2px solid #7e57c2;
}

QTabBar::tab:!selected:hover {
    background: #2a2a2a;
    color: #ddd;
}

QTabBar QToolButton {
    background: #252525;
    border-radius: 4px;
}

/* Checkboxes & Radio Buttons - Modern */
QCheckBox, QRadioButton {
    spacing: 10px;
    color: #e0e0e0;
}

QCheckBox::indicator, QRadioButton::indicator {
    width: 18px;
    height: 18px;
}

QCheckBox::indicator {
    border: 2px solid #666;
    border-radius: 4px;
    background: #1e1e1e;
}

QCheckBox::indicator:checked {
    background: #7e57c2;
    border-color: #7e57c2;
    image: url(:/icons/check_white.svg);
}

QRadioButton::indicator {
    border: 2px solid #666;
    border-radius: 9px;
    background: #1e1e1e;
}

QRadioButton::indicator:checked {
    background: #7e57c2;
    border-color: #7e57c2;
}

/* Pro Combo Boxes */
QComboBox {
    background: #1e1e1e;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 8px 12px;
    min-height: 36px;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 30px;
    border-left: 1px solid #333;
}

QComboBox::down-arrow {
    image: url(:/icons/chevron_down.svg);
    width: 16px;
    height: 16px;
}

QComboBox QAbstractItemView {
    background: #252525;
    border: 1px solid #333;
    selection-background-color: #7e57c2;
    padding: 8px 0;
    border-radius: 6px;
    outline: 0;
}

/* Enhanced Sliders */
QSlider::groove:horizontal {
    height: 4px;
    background: #333;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #f5f5f5;
    border: 1px solid #666;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #fff;
    border: 2px solid #7e57c2;
}

/* Progress Bars - Modern */
QProgressBar {
    background-color: #1e1e1e;
    border: 1px solid #333;
    border-radius: 6px;
    text-align: center;
    height: 20px;
}

QProgressBar::chunk {
    background-color: #7e57c2;
    border-radius: 5px;
    border-top-right-radius: 0;
    border-bottom-right-radius: 0;
}

/* Professional Scroll Bars */
QScrollBar:vertical, QScrollBar:horizontal {
    border: none;
    background: #1e1e1e;
    width: 10px;
    height: 10px;
    margin: 0;
}

QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #444;
    border-radius: 5px;
    min-height: 30px;
}

QScrollBar::handle:hover {
    background: #555;
}

QScrollBar::add-line, QScrollBar::sub-line {
    background: none;
    border: none;
}

/* Group Boxes - Refined */
QGroupBox {
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    margin-top: 20px;
    padding-top: 12px;
    font-weight: 500;
    color: #bbb;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    background-color: transparent;
    color: #e0e0e0;
}

/* Tables & Lists - Professional */
QTableView, QListView, QTreeView {
    background: #1e1e1e;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    gridline-color: #2a2a2a;
    alternate-background-color: #252525;
    outline: 0;
}

QHeaderView::section {
    background-color: #252525;
    color: #e0e0e0;
    padding: 8px;
    border: none;
    font-weight: 500;
}

/* Tooltips - Enhanced */
QToolTip {
    background-color: #252525;
    color: #e0e0e0;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 8px;
    opacity: 240;
}

/* Menus - Professional */
QMenuBar {
    background-color: #1a1a1a;
    padding: 4px;
    border-bottom: 1px solid #2a2a2a;
}

QMenuBar::item {
    padding: 6px 15px;
    border-radius: 4px;
}

QMenuBar::item:selected {
    background-color: #2a2a2a;
}

QMenu {
    background-color: #252525;
    border: 1px solid #333;
    padding: 6px;
    border-radius: 8px;
}

QMenu::item {
    padding: 8px 30px 8px 20px;
    border-radius: 4px;
}

QMenu::item:selected {
    background-color: #2a2a2a;
}

QMenu::separator {
    height: 1px;
    background: #333;
    margin: 6px 10px;
}

/* Toolbars - Refined */
QToolBar {
    background-color: #1a1a1a;
    border: none;
    border-bottom: 1px solid #2a2a2a;
    padding: 6px;
    spacing: 5px;
}

QToolBar::handle {
    width: 20px;
    image: url(:/icons/grip_dots.svg);
}

/* Status Bar - Professional */
QStatusBar {
    background-color: #1a1a1a;
    color: #aaa;
    border-top: 1px solid #2a2a2a;
    padding: 4px;
}

/* Dialogs - Enhanced */
QDialog {
    background-color: #1e1e1e;
    border: 1px solid #333;
    border-radius: 8px;
}

/* Spin Boxes - Modern */
QSpinBox, QDoubleSpinBox {
    background: #1e1e1e;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 6px;
    min-height: 36px;
}

QSpinBox::up-button, QDoubleSpinBox::up-button {
    width: 20px;
    border-left: 1px solid #333;
    border-top-right-radius: 6px;
}

QSpinBox::down-button, QDoubleSpinBox::down-button {
    width: 20px;
    border-left: 1px solid #333;
    border-bottom-right-radius: 6px;
}
QMainWindow::separator {
    background-color: #1a1a1a;
    width: 1px;
    height: 1px;
}
QToolBar + QSplitter::handle:horizontal {
    margin-top: -1px;
}
QSplitter::handle {
    background-color: #333;
    border: 1px solid #444;
}
QSplitter::handle:hover {
    background-color: #444;
    border-color: #7e57c2;
}
QSplitter::handle:horizontal { height: 9px; }
QSplitter::handle:vertical { width: 9px; }

            """
            
        elif theme == "light":
            stylesheet = """
            QMainWindow {
	background-color:#ececec;
}
QTextEdit {
	border-width: 1px;
	border-style: solid;
	border-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(0, 113, 255, 255), stop:1 rgba(91, 171, 252, 255));
}
QPlainTextEdit {
	border-width: 1px;
	border-style: solid;
	border-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(0, 113, 255, 255), stop:1 rgba(91, 171, 252, 255));
}
QToolButton {
	border-style: solid;
	border-top-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgb(215, 215, 215), stop:1 rgb(222, 222, 222));
	border-right-color: qlineargradient(spread:pad, x1:0, y1:0.5, x2:1, y2:0.5, stop:0 rgb(217, 217, 217), stop:1 rgb(227, 227, 227));
	border-left-color: qlineargradient(spread:pad, x1:0, y1:0.5, x2:1, y2:0.5, stop:0 rgb(227, 227, 227), stop:1 rgb(217, 217, 217));
	border-bottom-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgb(215, 215, 215), stop:1 rgb(222, 222, 222));
	border-width: 1px;
	border-radius: 5px;
	color: rgb(0,0,0);
	padding: 2px;
	background-color: rgb(255,255,255);
}
QToolButton:hover{
	border-style: solid;
	border-top-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgb(195, 195, 195), stop:1 rgb(222, 222, 222));
	border-right-color: qlineargradient(spread:pad, x1:0, y1:0.5, x2:1, y2:0.5, stop:0 rgb(197, 197, 197), stop:1 rgb(227, 227, 227));
	border-left-color: qlineargradient(spread:pad, x1:0, y1:0.5, x2:1, y2:0.5, stop:0 rgb(227, 227, 227), stop:1 rgb(197, 197, 197));
	border-bottom-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgb(195, 195, 195), stop:1 rgb(222, 222, 222));
	border-width: 1px;
	border-radius: 5px;
	color: rgb(0,0,0);
	padding: 2px;
	background-color: rgb(255,255,255);
}
QToolButton:pressed{
	border-style: solid;
	border-top-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgb(215, 215, 215), stop:1 rgb(222, 222, 222));
	border-right-color: qlineargradient(spread:pad, x1:0, y1:0.5, x2:1, y2:0.5, stop:0 rgb(217, 217, 217), stop:1 rgb(227, 227, 227));
	border-left-color: qlineargradient(spread:pad, x1:0, y1:0.5, x2:1, y2:0.5, stop:0 rgb(227, 227, 227), stop:1 rgb(217, 217, 217));
	border-bottom-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgb(215, 215, 215), stop:1 rgb(222, 222, 222));
	border-width: 1px;
	border-radius: 5px;
	color: rgb(0,0,0);
	padding: 2px;
	background-color: rgb(142,142,142);
}
QPushButton{
	border-style: solid;
	border-top-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgb(215, 215, 215), stop:1 rgb(222, 222, 222));
	border-right-color: qlineargradient(spread:pad, x1:0, y1:0.5, x2:1, y2:0.5, stop:0 rgb(217, 217, 217), stop:1 rgb(227, 227, 227));
	border-left-color: qlineargradient(spread:pad, x1:0, y1:0.5, x2:1, y2:0.5, stop:0 rgb(227, 227, 227), stop:1 rgb(217, 217, 217));
	border-bottom-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgb(215, 215, 215), stop:1 rgb(222, 222, 222));
	border-width: 1px;
	border-radius: 5px;
	color: rgb(0,0,0);
	padding: 2px;
	background-color: rgb(255,255,255);
}
QPushButton::default{
	border-style: solid;
	border-top-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgb(215, 215, 215), stop:1 rgb(222, 222, 222));
	border-right-color: qlineargradient(spread:pad, x1:0, y1:0.5, x2:1, y2:0.5, stop:0 rgb(217, 217, 217), stop:1 rgb(227, 227, 227));
	border-left-color: qlineargradient(spread:pad, x1:0, y1:0.5, x2:1, y2:0.5, stop:0 rgb(227, 227, 227), stop:1 rgb(217, 217, 217));
	border-bottom-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgb(215, 215, 215), stop:1 rgb(222, 222, 222));
	border-width: 1px;
	border-radius: 5px;
	color: rgb(0,0,0);
	padding: 2px;
	background-color: rgb(255,255,255);
}
QPushButton:hover{
	border-style: solid;
	border-top-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgb(195, 195, 195), stop:1 rgb(222, 222, 222));
	border-right-color: qlineargradient(spread:pad, x1:0, y1:0.5, x2:1, y2:0.5, stop:0 rgb(197, 197, 197), stop:1 rgb(227, 227, 227));
	border-left-color: qlineargradient(spread:pad, x1:0, y1:0.5, x2:1, y2:0.5, stop:0 rgb(227, 227, 227), stop:1 rgb(197, 197, 197));
	border-bottom-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgb(195, 195, 195), stop:1 rgb(222, 222, 222));
	border-width: 1px;
	border-radius: 5px;
	color: rgb(0,0,0);
	padding: 2px;
	background-color: rgb(255,255,255);
}
QPushButton:pressed{
	border-style: solid;
	border-top-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgb(215, 215, 215), stop:1 rgb(222, 222, 222));
	border-right-color: qlineargradient(spread:pad, x1:0, y1:0.5, x2:1, y2:0.5, stop:0 rgb(217, 217, 217), stop:1 rgb(227, 227, 227));
	border-left-color: qlineargradient(spread:pad, x1:0, y1:0.5, x2:1, y2:0.5, stop:0 rgb(227, 227, 227), stop:1 rgb(217, 217, 217));
	border-bottom-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgb(215, 215, 215), stop:1 rgb(222, 222, 222));
	border-width: 1px;
	border-radius: 5px;
	color: rgb(0,0,0);
	padding: 2px;
	background-color: rgb(142,142,142);
}
QPushButton:disabled{
	border-style: solid;
	border-top-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgb(215, 215, 215), stop:1 rgb(222, 222, 222));
	border-right-color: qlineargradient(spread:pad, x1:0, y1:0.5, x2:1, y2:0.5, stop:0 rgb(217, 217, 217), stop:1 rgb(227, 227, 227));
	border-left-color: qlineargradient(spread:pad, x1:0, y1:0.5, x2:1, y2:0.5, stop:0 rgb(227, 227, 227), stop:1 rgb(217, 217, 217));
	border-bottom-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgb(215, 215, 215), stop:1 rgb(222, 222, 222));
	border-width: 1px;
	border-radius: 5px;
	color: #808086;
	padding: 2px;
	background-color: rgb(142,142,142);
}
QLineEdit {
	border-width: 1px; border-radius: 4px;
	border-style: solid;
	border-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(0, 113, 255, 255), stop:1 rgba(91, 171, 252, 255));
}
QLabel {
	color: #000000;
}
QLCDNumber {
	color: rgb(0, 113, 255, 255);
}
QProgressBar {
	text-align: center;
	color: rgb(240, 240, 240);
	border-width: 1px; 
	border-radius: 10px;
	border-color: rgb(230, 230, 230);
	border-style: solid;
	background-color:rgb(207,207,207);
}
QProgressBar::chunk {
	background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(49, 147, 250, 255), stop:1 rgba(34, 142, 255, 255));
	border-radius: 10px;
}
QMenuBar {
	background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(207, 209, 207, 255), stop:1 rgba(230, 229, 230, 255));
}
QMenuBar::item {
	color: #000000;
  	spacing: 3px;
  	padding: 1px 4px;
	background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(207, 209, 207, 255), stop:1 rgba(230, 229, 230, 255));
}

QMenuBar::item:selected {
  	background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(0, 113, 255, 255), stop:1 rgba(91, 171, 252, 255));
	color: #FFFFFF;
}
QMenu::item:selected {
	border-style: solid;
	border-top-color: transparent;
	border-right-color: transparent;
	border-left-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(0, 113, 255, 255), stop:1 rgba(91, 171, 252, 255));
	border-bottom-color: transparent;
	border-left-width: 2px;
	color: #000000;
	padding-left:15px;
	padding-top:4px;
	padding-bottom:4px;
	padding-right:7px;
}
QMenu::item {
	border-style: solid;
	border-top-color: transparent;
	border-right-color: transparent;
	border-left-color: transparent;
	border-bottom-color: transparent;
	border-bottom-width: 1px;
	color: #000000;
	padding-left:17px;
	padding-top:4px;
	padding-bottom:4px;
	padding-right:7px;
}
QTabWidget {
	color:rgb(0,0,0);
	background-color:#000000;
}
QTabWidget::pane {
		border-color: rgb(223,223,223);
		background-color:rgb(226,226,226);
		border-style: solid;
		border-width: 2px;
    	border-radius: 6px;
}
QTabBar::tab:first {
	border-style: solid;
	border-left-width:1px;
	border-right-width:0px;
	border-top-width:1px;
	border-bottom-width:1px;
	border-top-color: rgb(209,209,209);
	border-left-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(209, 209, 209, 209), stop:1 rgba(229, 229, 229, 229));
	border-bottom-color: rgb(229,229,229);
	border-top-left-radius: 4px;
	border-bottom-left-radius: 4px;
	color: #000000;
	padding: 3px;
	margin-left:0px;
	background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(247, 247, 247, 255), stop:1 rgba(255, 255, 255, 255));
}
QTabBar::tab:last {
	border-style: solid;
	border-width:1px;
	border-top-color: rgb(209,209,209);
	border-left-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(209, 209, 209, 209), stop:1 rgba(229, 229, 229, 229));
	border-right-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(209, 209, 209, 209), stop:1 rgba(229, 229, 229, 229));
	border-bottom-color: rgb(229,229,229);
	border-top-right-radius: 4px;
	border-bottom-right-radius: 4px;
	color: #000000;
	padding: 3px;
	margin-left:0px;
	background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(247, 247, 247, 255), stop:1 rgba(255, 255, 255, 255));
}
QTabBar::tab {
	border-style: solid;
	border-top-width:1px;
	border-bottom-width:1px;
	border-left-width:1px;
	border-top-color: rgb(209,209,209);
	border-left-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(209, 209, 209, 209), stop:1 rgba(229, 229, 229, 229));
	border-bottom-color: rgb(229,229,229);
	color: #000000;
	padding: 3px;
	margin-left:0px;
	background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(247, 247, 247, 255), stop:1 rgba(255, 255, 255, 255));
}
QTabBar::tab:selected, QTabBar::tab:last:selected, QTabBar::tab:hover {
  	border-style: solid;
  	border-left-width:1px;
	border-right-color: transparent;
	border-top-color: rgb(209,209,209);
	border-left-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(209, 209, 209, 209), stop:1 rgba(229, 229, 229, 229));
	border-bottom-color: rgb(229,229,229);
	color: #FFFFFF;
	padding: 3px;
	margin-left:0px;
	background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(0, 113, 255, 255), stop:1 rgba(91, 171, 252, 255));
}

QTabBar::tab:selected, QTabBar::tab:first:selected, QTabBar::tab:hover {
  	border-style: solid;
  	border-left-width:1px;
  	border-bottom-width:1px;
  	border-top-width:1px;
	border-right-color: transparent;
	border-top-color: rgb(209,209,209);
	border-left-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(209, 209, 209, 209), stop:1 rgba(229, 229, 229, 229));
	border-bottom-color: rgb(229,229,229);
	color: #FFFFFF;
	padding: 3px;
	margin-left:0px;
	background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(0, 113, 255, 255), stop:1 rgba(91, 171, 252, 255));
}

QCheckBox {
	color: #000000;
	padding: 2px;
}
QCheckBox:disabled {
	color: #808086;
	padding: 2px;
}

QCheckBox:hover {
	border-radius:4px;
	border-style:solid;
	padding-left: 1px;
	padding-right: 1px;
	padding-bottom: 1px;
	padding-top: 1px;
	border-width:1px;
	border-color: transparent;
}
QCheckBox::indicator:checked {

	height: 10px;
	width: 10px;
	border-style:solid;
	border-width: 1px;
	border-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(0, 113, 255, 255), stop:1 rgba(91, 171, 252, 255));
	color: #000000;
	background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(0, 113, 255, 255), stop:1 rgba(91, 171, 252, 255));
}
QCheckBox::indicator:unchecked {

	height: 10px;
	width: 10px;
	border-style:solid;
	border-width: 1px;
	border-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(0, 113, 255, 255), stop:1 rgba(91, 171, 252, 255));
	color: #000000;
}
QRadioButton {
	color: 000000;
	padding: 1px;
}
QRadioButton::indicator:checked {
	height: 10px;
	width: 10px;
	border-style:solid;
	border-radius:5px;
	border-width: 1px;
	border-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(0, 113, 255, 255), stop:1 rgba(91, 171, 252, 255));
	color: #a9b7c6;
	background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(0, 113, 255, 255), stop:1 rgba(91, 171, 252, 255));
}
QRadioButton::indicator:!checked {
	height: 10px;
	width: 10px;
	border-style:solid;
	border-radius:5px;
	border-width: 1px;
	border-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(0, 113, 255, 255), stop:1 rgba(91, 171, 252, 255));
	color: #a9b7c6;
	background-color: transparent;
}
QStatusBar {
	color:#027f7f;
}
QSpinBox {
	border-style: solid;
	border-width: 1px;
	border-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(0, 113, 255, 255), stop:1 rgba(91, 171, 252, 255));
}
QDoubleSpinBox {
	border-style: solid;
	border-width: 1px;
	border-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(0, 113, 255, 255), stop:1 rgba(91, 171, 252, 255));
}
QTimeEdit {
	border-style: solid;
	border-width: 1px;
	border-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(0, 113, 255, 255), stop:1 rgba(91, 171, 252, 255));
}
QDateTimeEdit {
	border-style: solid;
	border-width: 1px;
	border-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(0, 113, 255, 255), stop:1 rgba(91, 171, 252, 255));
}
QDateEdit {
	border-style: solid;
	border-width: 1px;
	border-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(0, 113, 255, 255), stop:1 rgba(91, 171, 252, 255));
}

QToolBox {
	color: #a9b7c6;
	background-color:#000000;
}
QToolBox::tab {
	color: #a9b7c6;
	background-color:#000000;
}
QToolBox::tab:selected {
	color: #FFFFFF;
	background-color:#000000;
}
QScrollArea {
	color: #FFFFFF;
	background-color:#000000;
}
QSlider::groove:horizontal {
	height: 5px;
	background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(49, 147, 250, 255), stop:1 rgba(34, 142, 255, 255));
}
QSlider::groove:vertical {
	width: 5px;
	background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(49, 147, 250, 255), stop:1 rgba(34, 142, 255, 255));
}
QSlider::handle:horizontal {
	background: rgb(253,253,253);
	border-style: solid;
	border-width: 1px;
	border-color: rgb(207,207,207);
	width: 12px;
	margin: -5px 0;
	border-radius: 7px;
}
QSlider::handle:vertical {
	background: rgb(253,253,253);
	border-style: solid;
	border-width: 1px;
	border-color: rgb(207,207,207);
	height: 12px;
	margin: 0 -5px;
	border-radius: 7px;
}
QSlider::add-page:horizontal {
    background: rgb(181,181,181);
}
QSlider::add-page:vertical {
    background: rgb(181,181,181);
}
QSlider::sub-page:horizontal {
    background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(49, 147, 250, 255), stop:1 rgba(34, 142, 255, 255));
}
QSlider::sub-page:vertical {
    background-color: qlineargradient(spread:pad, y1:0.5, x1:1, y2:0.5, x2:0, stop:0 rgba(49, 147, 250, 255), stop:1 rgba(34, 142, 255, 255));
}
QScrollBar:horizontal {
	max-height: 20px;
	border: 1px transparent grey;
	margin: 0px 20px 0px 20px;
}
QScrollBar:vertical {
	max-width: 20px;
	border: 1px transparent grey;
	margin: 20px 0px 20px 0px;
}
QScrollBar::handle:horizontal {
	background: rgb(253,253,253);
	border-style: solid;
	border-width: 1px;
	border-color: rgb(207,207,207);
	border-radius: 7px;
	min-width: 25px;
}
QScrollBar::handle:horizontal:hover {
	background: rgb(253,253,253);
	border-style: solid;
	border-width: 1px;
	border-color: rgb(147, 200, 200);
	border-radius: 7px;
	min-width: 25px;
}
QScrollBar::handle:vertical {
	background: rgb(253,253,253);
	border-style: solid;
	border-width: 1px;
	border-color: rgb(207,207,207);
	border-radius: 7px;
	min-height: 25px;
}
QScrollBar::handle:vertical:hover {
	background: rgb(253,253,253);
	border-style: solid;
	border-width: 1px;
	border-color: rgb(147, 200, 200);
	border-radius: 7px;
	min-height: 25px;
}
QScrollBar::add-line:horizontal {
   border: 2px transparent grey;
   border-top-right-radius: 7px;
   border-bottom-right-radius: 7px;
   background: rgba(34, 142, 255, 255);
   width: 20px;
   subcontrol-position: right;
   subcontrol-origin: margin;
}
QScrollBar::add-line:horizontal:pressed {
   border: 2px transparent grey;
   border-top-right-radius: 7px;
   border-bottom-right-radius: 7px;
   background: rgb(181,181,181);
   width: 20px;
   subcontrol-position: right;
   subcontrol-origin: margin;
}
QScrollBar::add-line:vertical {
   border: 2px transparent grey;
   border-bottom-left-radius: 7px;
   border-bottom-right-radius: 7px;
   background: rgba(34, 142, 255, 255);
   height: 20px;
   subcontrol-position: bottom;
   subcontrol-origin: margin;
}
QScrollBar::add-line:vertical:pressed {
   border: 2px transparent grey;
   border-bottom-left-radius: 7px;
   border-bottom-right-radius: 7px;
   background: rgb(181,181,181);
   height: 20px;
   subcontrol-position: bottom;
   subcontrol-origin: margin;
}
QScrollBar::sub-line:horizontal {
   border: 2px transparent grey;
   border-top-left-radius: 7px;
   border-bottom-left-radius: 7px;
   background: rgba(34, 142, 255, 255);
   width: 20px;
   subcontrol-position: left;
   subcontrol-origin: margin;
}
QScrollBar::sub-line:horizontal:pressed {
   border: 2px transparent grey;
   border-top-left-radius: 7px;
   border-bottom-left-radius: 7px;
   background: rgb(181,181,181);
   width: 20px;
   subcontrol-position: left;
   subcontrol-origin: margin;
}
QScrollBar::sub-line:vertical {
   border: 2px transparent grey;
   border-top-left-radius: 7px;
   border-top-right-radius: 7px;
   background: rgba(34, 142, 255, 255);
   height: 20px;
   subcontrol-position: top;
   subcontrol-origin: margin;
}
QScrollBar::sub-line:vertical:pressed {
   border: 2px transparent grey;
   border-top-left-radius: 7px;
   border-top-right-radius: 7px;
   background: rgb(181,181,181);
   height: 20px;
   subcontrol-position: top;
   subcontrol-origin: margin;
}
QScrollBar::left-arrow:horizontal {
   border: 1px transparent grey;
   border-top-left-radius: 3px;
   border-bottom-left-radius: 3px;
   width: 6px;
   height: 6px;
   background: white;
}
QScrollBar::right-arrow:horizontal {
   border: 1px transparent grey;
   border-top-right-radius: 3px;
   border-bottom-right-radius: 3px;
   width: 6px;
   height: 6px;
   background: white;
}
QScrollBar::up-arrow:vertical {
   border: 1px transparent grey;
   border-top-left-radius: 3px;
   border-top-right-radius: 3px;
   width: 6px;
   height: 6px;
   background: white;
}
QScrollBar::down-arrow:vertical {
   border: 1px transparent grey;
   border-bottom-left-radius: 3px;
   border-bottom-right-radius: 3px;
   width: 6px;
   height: 6px;
   background: white;
}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
   background: none;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
   background: none;
}
QWidget:disabled {
    color: #808086;
    background-color: #d3d3d3;
}
            """
            
        elif theme == "New Mode":
            stylesheet = """
QMainWindow {
	background-color:#000000;
}
QDialog {
	background-color:#000000;
}
QColorDialog {
	background-color:#000000;
}
QTextEdit {
	background-color:#000000;
	color: #a9b7c6;
}
QPlainTextEdit {
	selection-background-color:#f39c12;
	background-color:#000000;
	border-style: solid;
	border-top-color: transparent;
	border-right-color: transparent;
	border-left-color: transparent;
	border-bottom-color: transparent;
	border-width: 1px;
	color: #a9b7c6;
}
QPushButton{
	border-style: solid;
	border-top-color: transparent;
	border-right-color: transparent;
	border-left-color: transparent;
	border-bottom-color: transparent;
	border-width: 1px;
	border-style: solid;
	color: #a9b7c6;
	padding: 2px;
	background-color: #000000;
}
QPushButton::default{
	border-style: solid;
	border-top-color: transparent;
	border-right-color: transparent;
	border-left-color: transparent;
	border-bottom-color: #e67e22;
	border-width: 1px;
	color: #a9b7c6;
	padding: 2px;
	background-color: #000000;
}
QPushButton:hover{
	border-style: solid;
	border-top-color: transparent;
	border-right-color: transparent;
	border-left-color: transparent;
	border-bottom-color: #e67e22;
	border-bottom-width: 1px;
	border-bottom-radius: 6px;
	border-style: solid;
	color: #FFFFFF;
	padding-bottom: 2px;
	background-color: #000000;
}
QPushButton:pressed{
	border-style: solid;
	border-top-color: transparent;
	border-right-color: transparent;
	border-left-color: transparent;
	border-bottom-color: #e67e22;
	border-bottom-width: 2px;
	border-bottom-radius: 6px;
	border-style: solid;
	color: #e67e22;
	padding-bottom: 1px;
	background-color: #000000;
}
QPushButton:disabled{
	border-style: solid;
	border-top-color: transparent;
	border-right-color: transparent;
	border-left-color: transparent;
	border-bottom-color: transparent;
	border-bottom-width: 2px;
	border-bottom-radius: 6px;
	border-style: solid;
	color: #808086;
	padding-bottom: 1px;
	background-color: #000000;
}
QToolButton {
	border-style: solid;
	border-top-color: transparent;
	border-right-color: transparent;
	border-left-color: transparent;
	border-bottom-color: #e67e22;
	border-bottom-width: 1px;
	border-style: solid;
	color: #a9b7c6;
	padding: 2px;
	background-color: #000000;
}
QToolButton:hover{
	border-style: solid;
	border-top-color: transparent;
	border-right-color: transparent;
	border-left-color: transparent;
	border-bottom-color: #e67e22;
	border-bottom-width: 2px;
	border-bottom-radius: 6px;
	border-style: solid;
	color: #FFFFFF;
	padding-bottom: 1px;
	background-color: #000000;
}
QLineEdit {
	border-width: 1px; border-radius: 4px;
	border-color: rgb(58, 58, 58);
	border-style: inset;
	padding: 0 8px;
	color: #a9b7c6;
	background:#000000;
	selection-background-color:#007b50;
	selection-color: #FFFFFF;
}
QLabel {
	color: #a9b7c6;
}
QLCDNumber {
	color: #e67e22;
}
QProgressBar {
	text-align: center;
	color: rgb(240, 240, 240);
	border-width: 1px; 
	border-radius: 10px;
	border-color: rgb(58, 58, 58);
	border-style: inset;
	background-color:#000000;
}
QProgressBar::chunk {
	background-color: #e67e22;
	border-radius: 5px;
}
QMenu{
	background-color:#000000;
}
QMenuBar {
	background:rgb(0, 0, 0);
	color: #a9b7c6;
}
QMenuBar::item {
  	spacing: 3px; 
	padding: 1px 4px;
  	background: transparent;
}
QMenuBar::item:selected { 
  	border-style: solid;
	border-top-color: transparent;
	border-right-color: transparent;
	border-left-color: transparent;
	border-bottom-color: #e67e22;
	border-bottom-width: 1px;
	border-bottom-radius: 6px;
	border-style: solid;
	color: #FFFFFF;
	padding-bottom: 0px;
	background-color: #000000;
}
QMenu::item:selected {
	border-style: solid;
	border-top-color: transparent;
	border-right-color: transparent;
	border-left-color: #e67e22;
	border-bottom-color: transparent;
	border-left-width: 2px;
	color: #FFFFFF;
	padding-left:15px;
	padding-top:4px;
	padding-bottom:4px;
	padding-right:7px;
	background-color:#000000;
}
QMenu::item {
	border-style: solid;
	border-top-color: transparent;
	border-right-color: transparent;
	border-left-color: transparent;
	border-bottom-color: transparent;
	border-bottom-width: 1px;
	border-style: solid;
	color: #a9b7c6;
	padding-left:17px;
	padding-top:4px;
	padding-bottom:4px;
	padding-right:7px;
	background-color:#000000;
}
QTabWidget {
	color:rgb(0,0,0);
	background-color:#000000;
}
QTabWidget::pane {
		border-color: rgb(77,77,77);
		background-color:#000000;
		border-style: solid;
		border-width: 1px;
    	border-radius: 6px;
}
QTabBar::tab {
	border-style: solid;
	border-top-color: transparent;
	border-right-color: transparent;
	border-left-color: transparent;
	border-bottom-color: transparent;
	border-bottom-width: 1px;
	border-style: solid;
	color: #808086;
	padding: 3px;
	margin-left:3px;
	background-color:#000000;
}
QTabBar::tab:selected, QTabBar::tab:last:selected, QTabBar::tab:hover {
  	border-style: solid;
	border-top-color: transparent;
	border-right-color: transparent;
	border-left-color: transparent;
	border-bottom-color: #e67e22;
	border-bottom-width: 2px;
	border-style: solid;
	color: #FFFFFF;
	padding-left: 3px;
	padding-bottom: 2px;
	margin-left:3px;
	background-color:#000000;
}

QCheckBox {
	color: #a9b7c6;
	padding: 2px;
}
QCheckBox:disabled {
	color: #808086;
	padding: 2px;
}

QCheckBox:hover {
	border-radius:4px;
	border-style:solid;
	padding-left: 1px;
	padding-right: 1px;
	padding-bottom: 1px;
	padding-top: 1px;
	border-width:1px;
	border-color: rgb(87, 97, 106);
	background-color:#000000;
}
QCheckBox::indicator:checked {

	height: 10px;
	width: 10px;
	border-style:solid;
	border-width: 1px;
	border-color: #e67e22;
	color: #a9b7c6;
	background-color: #e67e22;
}
QCheckBox::indicator:unchecked {

	height: 10px;
	width: 10px;
	border-style:solid;
	border-width: 1px;
	border-color: #e67e22;
	color: #a9b7c6;
	background-color: transparent;
}
QRadioButton {
	color: #a9b7c6;
	background-color:#000000;
	padding: 1px;
}
QRadioButton::indicator:checked {
	height: 10px;
	width: 10px;
	border-style:solid;
	border-radius:5px;
	border-width: 1px;
	border-color: #e67e22;
	color: #a9b7c6;
	background-color: #e67e22;
}
QRadioButton::indicator:!checked {
	height: 10px;
	width: 10px;
	border-style:solid;
	border-radius:5px;
	border-width: 1px;
	border-color: #e67e22;
	color: #a9b7c6;
	background-color: transparent;
}
QStatusBar {
	color:#027f7f;
}
QSpinBox {
	color: #a9b7c6;	
	background-color:#000000;
}
QDoubleSpinBox {
	color: #a9b7c6;	
	background-color:#000000;
}
QTimeEdit {
	color: #a9b7c6;	
	background-color:#000000;
}
QDateTimeEdit {
	color: #a9b7c6;	
	background-color:#000000;
}
QDateEdit {
	color: #a9b7c6;	
	background-color:#000000;
}
QComboBox {
	color: #a9b7c6;	
	background: #1e1d23;
}
QComboBox:editable {
	background: #1e1d23;
	color: #a9b7c6;
	selection-background-color:#000000;
}
QComboBox QAbstractItemView {
	color: #a9b7c6;	
	background: #1e1d23;
	selection-color: #FFFFFF;
	selection-background-color:#000000;
}
QComboBox:!editable:on, QComboBox::drop-down:editable:on {
	color: #a9b7c6;	
	background: #1e1d23;
}
QFontComboBox {
	color: #a9b7c6;	
	background-color:#000000;
}
QToolBox {
	color: #a9b7c6;
	background-color:#000000;
}
QToolBox::tab {
	color: #a9b7c6;
	background-color:#000000;
}
QToolBox::tab:selected {
	color: #FFFFFF;
	background-color:#000000;
}
QScrollArea {
	color: #FFFFFF;
	background-color:#000000;
}
QSlider::groove:horizontal {
	height: 5px;
	background: #e67e22;
}
QSlider::groove:vertical {
	width: 5px;
	background: #e67e22;
}
QSlider::handle:horizontal {
	background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
	border: 1px solid #5c5c5c;
	width: 14px;
	margin: -5px 0;
	border-radius: 7px;
}
QSlider::handle:vertical {
	background: qlineargradient(x1:1, y1:1, x2:0, y2:0, stop:0 #b4b4b4, stop:1 #8f8f8f);
	border: 1px solid #5c5c5c;
	height: 14px;
	margin: 0 -5px;
	border-radius: 7px;
}
QSlider::add-page:horizontal {
    background: white;
}
QSlider::add-page:vertical {
    background: white;
}
QSlider::sub-page:horizontal {
    background: #e67e22;
}
QSlider::sub-page:vertical {
    background: #e67e22;
}
QScrollBar:horizontal {
	max-height: 20px;
	background: rgb(0,0,0);
	border: 1px transparent grey;
	margin: 0px 20px 0px 20px;
}
QScrollBar::handle:horizontal {
	background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 rgba(255, 0, 0, 0), stop:0.7 rgba(255, 0, 0, 0), stop:0.71 rgb(230, 126, 34), stop:1 rgb(230, 126, 34));
	border-style: solid;
	border-width: 1px;
	border-color: rgb(0,0,0);
	min-width: 25px;
}
QScrollBar::handle:horizontal:hover {
	background: rgb(230, 126, 34);
	border-style: solid;
	border-width: 1px;
	border-color: rgb(0,0,0);
	min-width: 25px;
}
QScrollBar::add-line:horizontal {
  	border: 1px solid;
  	border-color: rgb(0,0,0);
  	background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 rgba(255, 0, 0, 0), stop:0.7 rgba(255, 0, 0, 0), stop:0.71 rgb(230, 126, 34), stop:1 rgb(230, 126, 34));
  	width: 20px;
  	subcontrol-position: right;
  	subcontrol-origin: margin;
}
QScrollBar::add-line:horizontal:hover {
  	border: 1px solid;
  	border-color: rgb(0,0,0);
	border-radius: 8px;
  	background: rgb(230, 126, 34);
  	height: 16px;
  	width: 16px;
  	subcontrol-position: right;
  	subcontrol-origin: margin;
}
QScrollBar::add-line:horizontal:pressed {
  	border: 1px solid;
  	border-color: grey;
	border-radius: 8px;
  	background: rgb(230, 126, 34);
  	height: 16px;
  	width: 16px;
  	subcontrol-position: right;
  	subcontrol-origin: margin;
}
QScrollBar::sub-line:horizontal {
  	border: 1px solid;
  	border-color: rgb(0,0,0);
  	background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 rgba(255, 0, 0, 0), stop:0.7 rgba(255, 0, 0, 0), stop:0.71 rgb(230, 126, 34), stop:1 rgb(230, 126, 34));
  	width: 20px;
  	subcontrol-position: left;
  	subcontrol-origin: margin;
}
QScrollBar::sub-line:horizontal:hover {
  	border: 1px solid;
  	border-color: rgb(0,0,0);
	border-radius: 8px;
  	background: rgb(230, 126, 34);
  	height: 16px;
  	width: 16px;
  	subcontrol-position: left;
  	subcontrol-origin: margin;
}
QScrollBar::sub-line:horizontal:pressed {
  	border: 1px solid;
  	border-color: grey;
	border-radius: 8px;
  	background: rgb(230, 126, 34);
  	height: 16px;
  	width: 16px;
  	subcontrol-position: left;
  	subcontrol-origin: margin;
}
QScrollBar::left-arrow:horizontal {
  	border: 1px transparent grey;
  	border-radius: 3px;
  	width: 6px;
  	height: 6px;
 	background: rgb(0,0,0);
}
QScrollBar::right-arrow:horizontal {
	border: 1px transparent grey;
	border-radius: 3px;
  	width: 6px;
  	height: 6px;
 	background: rgb(0,0,0);
}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
 	background: none;
} 
QScrollBar:vertical {
	max-width: 20px;
	background: rgb(0,0,0);
	border: 1px transparent grey;
	margin: 20px 0px 20px 0px;
}
QScrollBar::add-line:vertical {
	border: 1px solid;
  	border-color: rgb(0,0,0);
  	background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 rgba(255, 0, 0, 0), stop:0.7 rgba(255, 0, 0, 0), stop:0.71 rgb(230, 126, 34), stop:1 rgb(230, 126, 34));
  	height: 20px;
  	subcontrol-position: bottom;
  	subcontrol-origin: margin;
}
QScrollBar::add-line:vertical:hover {
  	border: 1px solid;
  	border-color: rgb(0,0,0);
	border-radius: 8px;
  	background: rgb(230, 126, 34);
  	height: 16px;
  	width: 16px;
  	subcontrol-position: bottom;
  	subcontrol-origin: margin;
}
QScrollBar::add-line:vertical:pressed {
  	border: 1px solid;
  	border-color: grey;
	border-radius: 8px;
  	background: rgb(230, 126, 34);
  	height: 16px;
  	width: 16px;
  	subcontrol-position: bottom;
  	subcontrol-origin: margin;
}
QScrollBar::sub-line:vertical {
  	border: 1px solid;
  	border-color: rgb(0,0,0);
	background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 rgba(255, 0, 0, 0), stop:0.7 rgba(255, 0, 0, 0), stop:0.71 rgb(230, 126, 34), stop:1 rgb(230, 126, 34));
  	height: 20px;
  	subcontrol-position: top;
  	subcontrol-origin: margin;
}
QScrollBar::sub-line:vertical:hover {
  	border: 1px solid;
  	border-color: rgb(0,0,0);
	border-radius: 8px;
  	background: rgb(230, 126, 34);
  	height: 16px;
  	width: 16px;
  	subcontrol-position: top;
  	subcontrol-origin: margin;
}
QScrollBar::sub-line:vertical:pressed {
  	border: 1px solid;
  	border-color: grey;
	border-radius: 8px;
  	background: rgb(230, 126, 34);
  	height: 16px;
  	width: 16px;
  	subcontrol-position: top;
  	subcontrol-origin: margin;
}
	QScrollBar::handle:vertical {
	background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 rgba(255, 0, 0, 0), stop:0.7 rgba(255, 0, 0, 0), stop:0.71 rgb(230, 126, 34), stop:1 rgb(230, 126, 34));
	border-style: solid;
	border-width: 1px;
	border-color: rgb(0,0,0);
	min-height: 25px;
}
QScrollBar::handle:vertical:hover {
	background: rgb(230, 126, 34);
	border-style: solid;
	border-width: 1px;
	border-color: rgb(0,0,0);
	min-heigth: 25px;
}
QScrollBar::up-arrow:vertical {
	border: 1px transparent grey;
	border-radius: 3px;
  	width: 6px;
  	height: 6px;
 	background: rgb(0,0,0);
}
QScrollBar::down-arrow:vertical {
  	border: 1px transparent grey;
  	border-radius: 3px;
  	width: 6px;
  	height: 6px;
 	background: rgb(0,0,0);
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
  	background: none;
}

            """
        else:
            stylesheet = ""
        
        self.setStyleSheet(stylesheet)
        logger.info(f"Applied {theme} theme")