import sys
import subprocess
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout,
                             QWidget, QLabel, QPushButton, QTextEdit,
                             QScrollArea, QMessageBox)
from PyQt5.QtCore import QProcess, Qt
from PyQt5.QtGui import QColor, QFont


class EmailAssistantGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.check_dependencies()

    def setup_ui(self):
        self.setWindowTitle("Voice Email Assistant")
        self.setGeometry(100, 100, 900, 700)

        # Main Widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()

        # Header
        header = QLabel("Voice Email Assistant")
        header.setFont(QFont('Arial', 16, QFont.Bold))
        header.setStyleSheet("color: #2b5b84; padding: 10px;")
        layout.addWidget(header)

        # Console Output
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setFont(QFont('Consolas', 10))
        self.console.setStyleSheet("""
            background-color: #f8f9fa;
            color: #212529;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 8px;
        """)

        # Scroll Area
        scroll = QScrollArea()
        scroll.setWidget(self.console)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll, stretch=1)

        # Control Buttons
        self.install_btn = QPushButton("Install Dependencies")
        self.install_btn.setStyleSheet(self.get_button_style("#17a2b8"))
        self.install_btn.clicked.connect(self.install_dependencies)

        self.start_btn = QPushButton("Start Assistant")
        self.start_btn.setStyleSheet(self.get_button_style("#28a745"))
        self.start_btn.clicked.connect(self.start_assistant)
        self.start_btn.setEnabled(False)

        self.stop_btn = QPushButton("Stop Assistant")
        self.stop_btn.setStyleSheet(self.get_button_style("#dc3545"))
        self.stop_btn.clicked.connect(self.stop_assistant)
        self.stop_btn.setEnabled(False)

        # Button Layout
        btn_layout = QVBoxLayout()
        btn_layout.addWidget(self.install_btn)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)

        # Process handler
        self.process = QProcess()
        self.process.readyReadStandardOutput.connect(self.handle_output)
        self.process.readyReadStandardError.connect(self.handle_error)
        self.process.finished.connect(self.process_finished)

        main_widget.setLayout(layout)

    def get_button_style(self, color):
        return f"""
            QPushButton {{
                background-color: {color};
                color: white;
                font-weight: bold;
                padding: 12px;
                border-radius: 6px;
                min-width: 200px;
                margin: 5px;
            }}
            QPushButton:hover {{
                background-color: #{hex(int(color[1:], 16) - 0x222222)[2:]};
            }}
            QPushButton:disabled {{
                background-color: #6c757d;
            }}
        """

    def check_dependencies(self):
        """Check if required packages are installed"""
        self.console.append("Checking dependencies...")
        missing = []

        required = [
            'speech_recognition',
            'pyttsx3',
            'sklearn',
            'transformers',
            'torch',
            'PIL',
            'imaplib2'
        ]

        for package in required:
            try:
                __import__(package)
                self.console.append(f"✓ {package} installed")
            except ImportError:
                missing.append(package)
                self.console.append(f"✗ {package} missing")

        if missing:
            self.console.append(
                "\n<font color='red'>Some dependencies are missing. Click 'Install Dependencies'</font>")
            self.install_btn.setEnabled(True)
            self.start_btn.setEnabled(False)
        else:
            self.console.append("\n<font color='green'>All dependencies are installed!</font>")
            self.install_btn.setEnabled(False)
            self.start_btn.setEnabled(True)

    def install_dependencies(self):
        """Install required packages"""
        self.console.append("\nStarting installation...")
        QApplication.processEvents()

        try:
            # Run the installation script
            process = subprocess.Popen(
                [sys.executable, "install_requirement.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    self.console.append(output.strip())

            # Check installation result
            if process.returncode == 0:
                self.console.append("\n<font color='green'>Installation successful!</font>")
                self.check_dependencies()
            else:
                error = process.stderr.read()
                self.console.append(f"\n<font color='red'>Installation failed:\n{error}</font>")

        except Exception as e:
            self.console.append(f"\n<font color='red'>Error during installation:\n{str(e)}</font>")

    def start_assistant(self):
        """Start the assistant backend"""
        self.console.append("\n<font color='green'>Starting email assistant...</font>")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        # Start your main Python file
        self.process.start(sys.executable, ["voice_email_assistant.py"])

    def stop_assistant(self):
        """Stop the assistant backend"""
        self.console.append("\n<font color='orange'>Stopping assistant...</font>")
        self.process.terminate()

    def handle_output(self):
        """Capture and display stdout"""
        output = self.process.readAllStandardOutput().data().decode().strip()
        if output:
            self.console.append(output)
            self.scroll_to_bottom()

    def handle_error(self):
        """Capture and display stderr"""
        error = self.process.readAllStandardError().data().decode().strip()
        if error:
            self.console.append(f"<font color='red'>{error}</font>")
            self.scroll_to_bottom()

    def process_finished(self):
        """Handle process completion"""
        self.console.append("\n<font color='blue'>Assistant stopped.</font>")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def scroll_to_bottom(self):
        """Auto-scroll console to bottom"""
        scrollbar = self.console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle("Fusion")

    window = EmailAssistantGUI()
    window.show()
    sys.exit(app.exec_())