import subprocess
import sys

REQUIRED_PACKAGES = [
    'PyQt5',
    'speechrecognition',
    'pyttsx3',
    'python-dotenv',
    'scikit-learn',
    'transformers',
    'torch',
    'pillow',
    'imaplib2'
]

def install_packages():
    print("Installing required packages...")
    for package in REQUIRED_PACKAGES:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}")

if __name__ == "__main__":
    install_packages()
    print("\nAll packages installed. You can now run the application.")
