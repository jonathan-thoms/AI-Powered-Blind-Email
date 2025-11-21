# 📧 Voice-Activated AI Email Assistant

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green)
![AI](https://img.shields.io/badge/AI-Transformers%20%7C%20BLIP-orange)

A specialized desktop application designed to make email accessible for the visually impaired. This application uses **Speech Recognition**, **Text-to-Speech**, and **Generative AI** to read emails, summarize long content, and describe attached images aloud.

## 🚀 Features

-   **🗣️ Voice Control:** Navigate the inbox, read, and search emails completely hands-free.
-   **🧠 AI Summarization:** Uses `facebook/bart-large-cnn` to condense long corporate emails into brief summaries.
-   **👁️ Image Captioning:** Automatically detects images in emails and describes them using the Salesforce `BLIP` model (e.g., *"A photo of a group of people in a meeting room"*).
-   **🛡️ Spam & Priority Filter:** Uses a Naive Bayes classifier to flag spam and highlight urgent emails.
-   **🖥️ Interactive GUI:** A clean PyQt5 dashboard to monitor backend processes and logs.

## 🛠️ Tech Stack

-   **Core:** Python 3
-   **GUI:** PyQt5
-   **AI/ML:** HuggingFace Transformers, PyTorch, Scikit-Learn
-   **Audio:** SpeechRecognition, Pyttsx3
-   **Protocols:** IMAP (with SSL)

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/voice-email-assistant.git](https://github.com/yourusername/voice-email-assistant.git)
    cd voice-email-assistant
    ```

2.  **Set up Environment Variables:**
    Create a file named `.env` in the root directory and add your credentials:
    ```ini
    EMAIL_USER=your_email@gmail.com
    EMAIL_PASS=your_app_password
    ```
    *> **Note:** For Gmail, you must use an [App Password](https://support.google.com/accounts/answer/185833), not your login password.*

3.  **Install Dependencies:**
    You can use the built-in "Install Dependencies" button in the GUI, or run:
    ```bash
    pip install -r requirements.txt
    ```

## 📖 Usage

1.  Run the application:
    ```bash
    python gui.py
    ```
2.  Click **"Start Assistant"**.
3.  Wait for the "Voice Email Assistant Online" voice prompt.
4.  **Try these voice commands:**
    -   *"Check email"*
    -   *"Summarize email one"*
    -   *"Describe images in email two"*
    -   *"Search for 'meeting'"*
    -   *"Exit"*

## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
