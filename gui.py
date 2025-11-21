import sys
import subprocess
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout,
                             QWidget, QLabel, QPushButton, QTextEdit,
                             QScrollArea)
from PyQt5.QtCore import QProcess, Qt
from PyQt5.QtGui import QFont

class EmailAssistantGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.check_dependencies()
        self.process = None

    def setup_ui(self):
        self.setWindowTitle("Voice Email Assistant")
        self.setGeometry(100, 100, 900, 700)

        # Main container
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()

        # Header
        header = QLabel("AI Voice Email Assistant")
        header.setFont(QFont('Segoe UI', 18, QFont.Bold))
        header.setStyleSheet("color: #2c3e50; padding: 15px;")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        # Console Output
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setFont(QFont('Consolas', 10))
        self.console.setStyleSheet("""
            background-color: #1e1e1e;
            color: #00ff00;
            border-radius: 5px;
            padding: 10px;
        """)
        
        scroll = QScrollArea()
        scroll.setWidget(self.console)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        # Buttons
        btn_layout = QVBoxLayout()
        
        self.install_btn = self._create_btn("Install Dependencies", "#3498db")
        self.install_btn.clicked.connect(self.install_dependencies)
        
        self.start_btn = self._create_btn("Start Assistant", "#2ecc71")
        self.start_btn.clicked.connect(self.start_assistant)
        self.start_btn.setEnabled(False)
        
        self.stop_btn = self._create_btn("Stop Assistant", "#e74c3c")
        self.stop_btn.clicked.connect(self.stop_assistant)
        self.stop_btn.setEnabled(False)

        btn_layout.addWidget(self.install_btn)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)

        main_widget.setLayout(layout)

        # Process Handler
        self.process = QProcess()
        self.process.readyReadStandardOutput.connect(self.handle_output)
        self.process.readyReadStandardError.connect(self.handle_error)
        self.process.finished.connect(self.process_finished)

    def _create_btn(self, text, color):
        btn = QPushButton(text)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                font-weight: bold;
                padding: 12px;
                border-radius: 5px;
                font-size: 14px;
            }}
            QPushButton:hover {{ opacity: 0.8; }}
            QPushButton:disabled {{ background-color: #95a5a6; }}
        """)
        return btn

    def check_dependencies(self):
        self.log("Checking environment...")
        try:
            import speech_recognition
            import pyttsx3
            import torch
            self.log("Dependencies found.", "green")
            self.start_btn.setEnabled(True)
            self.install_btn.setEnabled(False)
        except ImportError:
            self.log("Dependencies missing. Please click Install.", "red")
            self.start_btn.setEnabled(False)
            self.install_btn.setEnabled(True)

    def install_dependencies(self):
        self.log("Installing dependencies from requirements.txt...")
        self.install_btn.setEnabled(False)
        QApplication.processEvents()
        
        # Run pip install -r requirements.txt
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            self.log("Installation Complete!", "green")
            self.check_dependencies()
        except subprocess.CalledProcessError as e:
            self.log(f"Installation Failed: {e}", "red")
            self.install_btn.setEnabled(True)

    def start_assistant(self):
        self.log("Starting Backend...", "blue")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # Robustly find the backend file
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend.py")
        self.process.start(sys.executable, [script_path])

    def stop_assistant(self):
        if self.process.state() == QProcess.Running:
            self.process.terminate()
            self.log("Stopping...", "orange")

    def handle_output(self):
        data = self.process.readAllStandardOutput().data().decode().strip()
        if data: self.log(data)

    def handle_error(self):
        data = self.process.readAllStandardError().data().decode().strip()
        if data: self.log(data, "red")

    def process_finished(self):
        self.log("Assistant Process Terminated.", "orange")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def log(self, message, color="white"):
        self.console.append(f"<font color='{color}'>{message}</font>")
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = EmailAssistantGUI()
    window.show()
    sys.exit(app.exec_())
