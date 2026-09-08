from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QTabWidget, QLabel, QLineEdit, QFileDialog, QDockWidget,
    QMessageBox, QGraphicsDropShadowEffect
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap, QColor

def add_shadow_effect(widget):
    shadow = QGraphicsDropShadowEffect()
    shadow.setBlurRadius(8)
    shadow.setColor(QColor(0, 0, 0, 50))
    shadow.setOffset(2, 2)
    widget.setGraphicsEffect(shadow)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hệ thống điểm danh AI")
        self.setGeometry(100, 100, 1280, 720)

        # Biến trạng thái
        self.current_class_id = None
        self.current_timetable_id = None
        self.recognized_students = set()
        self.student_map = {}
        self.recognition_active = False
        self.frame_count = 0
        self.sidebar_collapsed = False
        self.cap = None

        # Style cho button
        self.button_style = """
            QPushButton {
                background-color: #3b82f6; color: white; padding: 14px 18px;
                border-radius: 10px; font-size: 16.8px; font-weight: 500;
                text-align: left; border: none; margin-bottom: 10px;
            }
            QPushButton:hover { background-color: #2563eb; }
        """

        # Giao diện chính
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        # Sidebar
        self.setup_sidebar()

        # Main content
        self.main = QWidget()
        self.main_layout.addWidget(self.main)
        main_layout = QVBoxLayout(self.main)
        main_layout.setContentsMargins(20, 40, 20, 40)
        main_layout.setSpacing(0)
        self.main.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #e0f2fe, stop:1 #ffffff);
        """)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: none; }
            QTabBar::tab { background: #e0f2fe; padding: 10px; border-radius: 5px; }
            QTabBar::tab:selected { background: #2563eb; color: white; }
        """)
        main_layout.addWidget(self.tabs)

        # Tab Upload
        self.setup_upload_tab()

        # Tab Schedule
        self.setup_schedule_tab()

        # Tab Attendance
        self.setup_attendance_tab()

        # Timer cho nhận diện khuôn mặt
        self.timer = QTimer()

    def setup_sidebar(self):
        self.sidebar = QDockWidget("Menu", self)
        self.sidebar.setFixedWidth(280)
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(24, 24, 24, 24)
        sidebar_layout.setSpacing(10)

        self.nav_upload = QPushButton("📤 Upload ảnh")
        self.nav_schedule = QPushButton("📅 Thời khóa biểu")
        self.toggle_sidebar_btn = QPushButton("☰")
        self.toggle_sidebar_btn.setFixedWidth(40)
        self.toggle_sidebar_btn.setStyleSheet("padding: 10px; border-radius: 5px; background-color: #2563eb; color: white;")

        self.nav_upload.setStyleSheet(self.button_style)
        self.nav_schedule.setStyleSheet(self.button_style)
        title_label = QLabel("<h4>📚 Menu</h4>")
        title_label.setStyleSheet("""
            font-size: 25.6px; font-weight: 700; margin-bottom: 32px; color: white; letter-spacing: 0.5px;
        """)
        sidebar_layout.addWidget(title_label)
        sidebar_layout.addWidget(self.nav_upload)
        sidebar_layout.addWidget(self.nav_schedule)
        sidebar_layout.addWidget(self.toggle_sidebar_btn)
        sidebar_layout.addStretch()
        self.sidebar.setWidget(sidebar_widget)
        self.sidebar.setStyleSheet("""
            QDockWidget { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1e3a8a, stop:1 #1e40af); border: none; }
            QDockWidget::title { text-align: center; color: white; font-size: 18px; }
        """)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.sidebar)

    def setup_upload_tab(self):
        self.upload_tab = QWidget()
        upload_layout = QVBoxLayout(self.upload_tab)
        self.title_label_upload = QLabel("📤 Upload ảnh sinh viên")
        self.title_label_upload.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label_upload.setStyleSheet("""
            font-size: 30.4px; font-weight: 700; color: #1e3a8a; margin-bottom: 28.8px; letter-spacing: 0.3px;
        """)
        upload_layout.addWidget(self.title_label_upload)

        self.student_code_input = QLineEdit()
        self.student_code_input.setPlaceholderText("Mã sinh viên")
        self.student_code_input.setStyleSheet("""
            QLineEdit {
                border-radius: 10px; border: 1px solid #d1d5db; padding: 12px;
                font-size: 16px; background-color: #fff;
            }
            QLineEdit:focus { border-color: #2563eb; outline: none; }
        """)
        upload_layout.addWidget(self.student_code_input)

        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("Ảnh khuôn mặt")
        self.file_input.setReadOnly(True)
        self.file_input.setStyleSheet("""
            QLineEdit {
                border-radius: 10px; border: 1px solid #d1d5db; padding: 12px;
                font-size: 16px; background-color: #fff;
            }
        """)
        upload_layout.addWidget(self.file_input)

        self.browse_button = QPushButton("Chọn tệp")
        self.browse_button.setStyleSheet("""
            QPushButton {
                background-color: #2563eb; border-radius: 10px; padding: 12px 24px;
                font-weight: 600; font-size: 16px; border: none;
            }
            QPushButton:hover { background-color: #1e40af; }
        """)
        add_shadow_effect(self.browse_button)
        upload_layout.addWidget(self.browse_button)

        self.upload_button = QPushButton("Gửi ảnh")
        self.upload_button.setStyleSheet("""
            QPushButton {
                background-color: #2563eb; border-radius: 10px; padding: 12px 24px;
                font-weight: 600; font-size: 16px; border: none;
            }
            QPushButton:hover { background-color: #1e40af; }
        """)
        add_shadow_effect(self.upload_button)
        upload_layout.addWidget(self.upload_button)

        self.upload_result = QLabel("Kết quả tải ảnh: Chưa tải")
        self.upload_result.setStyleSheet("""
            font-weight: 500; border-radius: 10px; background-color: #dcfce7;
            color: #166534; padding: 16px 24px; margin-top: 16px;
        """)
        upload_layout.addWidget(self.upload_result)
        upload_layout.addStretch()
        self.tabs.addTab(self.upload_tab, "Upload ảnh")

    def setup_schedule_tab(self):
        self.schedule_tab = QWidget()
        schedule_layout = QVBoxLayout(self.schedule_tab)
        self.title_label_schedule = QLabel("📅 Thời khóa biểu")
        self.title_label_schedule.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label_schedule.setStyleSheet("""
            font-size: 30.4px; font-weight: 700; color: #1e3a8a; margin-bottom: 28.8px; letter-spacing: 0.3px;
        """)
        schedule_layout.addWidget(self.title_label_schedule)

        self.schedule_table = QTableWidget()
        self.schedule_table.setStyleSheet("""
            QTableWidget {
                border-radius: 14px; background-color: #fff; border: 1px solid #e5e7eb;
            }
            QTableWidget::item {
                padding: 16px; font-size: 15.2px; font-weight: 500; text-align: center; border-bottom: 1px solid #e5e7eb;
            }
            QTableWidget::item:last-child { border-bottom: none; }
            QHeaderView::section {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1e3a8a, stop:1 #2563eb);
                color: white; padding: 16px; font-weight: 600; font-size: 16px; border: none;
            }
        """)
        schedule_layout.addWidget(self.schedule_table)
        self.tabs.addTab(self.schedule_tab, "Thời khóa biểu")

    def setup_attendance_tab(self):
        self.attendance_tab = QWidget()
        attendance_layout = QHBoxLayout(self.attendance_tab)
        self.video_container = QWidget()
        self.video_container.setFixedSize(1280, 720)
        self.video_container.setStyleSheet("""
            border-radius: 20px; background-color: #111827; margin-top: 28px;
        """)
        self.video_container.setVisible(False)
        video_layout = QVBoxLayout(self.video_container)
        self.video_label = QLabel()
        video_layout.addWidget(self.video_label)

        attendance_right_layout = QVBoxLayout()
        self.subject_name = QLabel("Môn học: Chưa chọn")
        self.class_name = QLabel("Lớp: Chưa chọn")
        self.subject_name.setStyleSheet("font-size: 30.4px; color: #1e3a8a; font-weight: 700;")
        self.class_name.setStyleSheet("font-size: 30.4px; color: #1e3a8a; font-weight: 700;")
        attendance_right_layout.addWidget(self.subject_name)
        attendance_right_layout.addWidget(self.class_name)

        self.student_table = QTableWidget()
        self.student_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #e5e7eb; border-radius: 10px; background-color: #fff;
            }
            QHeaderView::section { background: #1e3a8a; color: white; padding: 16px; font-size: 16px; }
            QTableWidget::item { text-align: center; padding: 16px; }
        """)
        attendance_right_layout.addWidget(self.student_table)

        self.start_recognition_btn = QPushButton("🤳 Điểm danh bằng khuôn mặt")
        self.start_recognition_btn.setStyleSheet("""
            QPushButton {
                background-color: #0ea5e9; border-radius: 10px; padding: 12px 24px;
                font-weight: 600; font-size: 16px; border: none;
            }
            QPushButton:hover { background-color: #0284c7; }
        """)
        add_shadow_effect(self.start_recognition_btn)
        attendance_right_layout.addWidget(self.start_recognition_btn)

        self.save_attendance_btn = QPushButton("💾 Lưu điểm danh")
        self.save_attendance_btn.setStyleSheet("""
            QPushButton {
                background-color: #16a34a; border-radius: 10px; padding: 12px 24px;
                font-weight: 600; font-size: 16px; border: none;
            }
            QPushButton:hover { background-color: #15803d; }
        """)
        add_shadow_effect(self.save_attendance_btn)
        attendance_right_layout.addWidget(self.save_attendance_btn)

        attendance_layout.addWidget(self.video_container)
        attendance_layout.addLayout(attendance_right_layout)
        self.tabs.addTab(self.attendance_tab, "Điểm danh")

    def toggle_sidebar(self):
        self.sidebar_collapsed = not self.sidebar_collapsed
        if self.sidebar_collapsed:
            self.sidebar.setFixedWidth(160)
            self.nav_upload.setText("📤")
            self.nav_schedule.setText("📅")
            self.toggle_sidebar_btn.setText("▶")
            self.nav_upload.setStyleSheet(self.button_style + "text-align: center; padding: 14px 10px;")
            self.nav_schedule.setStyleSheet(self.button_style + "text-align: center; padding: 14px 10px;")
        else:
            self.sidebar.setFixedWidth(280)
            self.nav_upload.setText("📤 Upload ảnh")
            self.nav_schedule.setText("📅 Thời khóa biểu")
            self.nav_upload.setStyleSheet(self.button_style)
            self.nav_schedule.setStyleSheet(self.button_style)
            self.toggle_sidebar_btn.setText("☰")

    def fetch_schedule(self):
        pass  # Được triển khai trong main.py

    def update_frame(self):
        pass  # Được triển khai trong main.py