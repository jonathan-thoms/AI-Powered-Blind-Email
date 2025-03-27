import imaplib
import email
import re
import speech_recognition as sr
import pyttsx3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from transformers import pipeline  # For summarization
import numpy as np


class VoiceEmailManager:
    def __init__(self):
        # Email Credentials
        self.EMAIL_USER = "emailtest.jonathan@gmail.com"
        self.EMAIL_PASS = "xwis udwz bwgr ueuh"

        # Voice Engine
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 150)
        self.recognizer = sr.Recognizer()

        # Spam Filter Model (Naive Bayes)
        self.vectorizer = TfidfVectorizer()
        self.spam_model = MultinomialNB()
        self._train_dummy_spam_model()  # Initialize with dummy data

        # Summarization Model (BART)
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

        # Priority Detection Rules
        self.priority_senders = ["important@domain.com", "boss@company.com"]
        self.priority_keywords = ["urgent", "meeting", "deadline"]

        # Number Mapping
        self.number_words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
        }

    # --- Core Functions (Existing Code) ---
    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def listen_command(self):
        with sr.Microphone() as source:
            self.speak("Listening...")
            self.recognizer.adjust_for_ambient_noise(source)
            try:
                audio = self.recognizer.listen(source, timeout=5)
                return self.recognizer.recognize_google(audio).lower()
            except (sr.UnknownValueError, sr.RequestError):
                self.speak("Sorry, I couldn't understand. Try again.")
                return None

    def fetch_emails(self, folder="inbox", unseen=True):
        try:
            mail = imaplib.IMAP4_SSL("imap.gmail.com")
            mail.login(self.EMAIL_USER, self.EMAIL_PASS)
            mail.select(folder)

            status, messages = mail.search(None, "UNSEEN" if unseen else "ALL")
            email_ids = messages[0].split()

            email_list = {}
            for idx, email_id in enumerate(email_ids[:5], start=1):
                _, msg_data = mail.fetch(email_id, "(RFC822)")
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        email_list[idx] = (email_id, msg)
            mail.logout()
            return email_list
        except Exception as e:
            self.speak("Error fetching emails.")
            print(str(e))
            return {}

    # --- New Features to Add ---
    # Feature 1: Spam Detection (Naive Bayes)
    def _train_dummy_spam_model(self):
        # Replace with real training data later
        X_train = ["win free money", "meeting at 3pm", "urgent project update"]
        y_train = [1, 0, 0]  # 1=spam, 0=ham
        X_vec = self.vectorizer.fit_transform(X_train)
        self.spam_model.fit(X_vec, y_train)

    def is_spam(self, email_text):
        X_vec = self.vectorizer.transform([email_text])
        return self.spam_model.predict(X_vec)[0] == 1

    # Feature 2: Priority Detection
    def is_priority(self, email_msg):
        subject = email_msg["subject"] or ""
        sender = email_msg["from"] or ""

        # Rule 1: Sender is in priority list
        if any(priority_sender in sender for priority_sender in self.priority_senders):
            return True

        # Rule 2: Keywords in subject/body
        body = self._get_email_body(email_msg)
        if any(keyword in subject.lower() or keyword in body.lower()
               for keyword in self.priority_keywords):
            return True

        return False

    # Feature 3: Summarization
    def summarize_email(self, email_msg, max_length=150):
        body = self._get_email_body(email_msg)
        summary = self.summarizer(body, max_length=max_length, min_length=30, do_sample=False)
        return summary[0]["summary_text"]

    # Feature 4: Voice-Activated Search
    def search_emails(self, query, folder="inbox"):
        emails = self.fetch_emails(folder, unseen=False)
        results = []
        for idx, (email_id, msg) in emails.items():
            body = self._get_email_body(msg)
            if query.lower() in body.lower() or query.lower() in msg["subject"].lower():
                results.append((idx, msg))
        return results

    # --- Helper Functions ---
    def _get_email_body(self, msg):
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    return part.get_payload(decode=True).decode("utf-8")
        else:
            return msg.get_payload(decode=True).decode("utf-8")

    def extract_number(self, command):
        match = re.search(r"(\d+)", command)
        if match:
            return int(match.group(1))
        for word in command.split():
            if word in self.number_words:
                return self.number_words[word]
        return None

    # --- Main Loop ---
    def run(self):
        self.speak("Welcome to your voice-controlled email assistant.")
        while True:
            self.speak("Say 'check email', 'search emails', or 'exit'.")
            command = self.listen_command()

            if command == "check email":
                self._handle_email_checking()
            elif command.startswith("search"):
                query = command.replace("search", "").strip()
                self._handle_email_search(query)
            elif command == "exit":
                self.speak("Goodbye!")
                break

    def _handle_email_checking(self):
        emails = self.fetch_emails()
        if emails:
            for idx, (email_id, msg) in emails.items():
                sender = msg["from"]
                subject = msg["subject"]
                priority_flag = " (Priority)" if self.is_priority(msg) else ""
                spam_flag = " (Spam)" if self.is_spam(subject + " " + self._get_email_body(msg)) else ""

                self.speak(f"Email {idx}: From {sender}. Subject: {subject}{priority_flag}{spam_flag}")

            self.speak("Say 'open X' to read an email, or 'summarize X'.")
            while True:
                cmd = self.listen_command()
                if cmd and ("open" in cmd or "summarize" in cmd):
                    num = self.extract_number(cmd)
                    if num in emails:
                        if "summarize" in cmd:
                            summary = self.summarize_email(emails[num][1])
                            self.speak(f"Summary: {summary}")
                        else:
                            self.speak(self._get_email_body(emails[num][1]))
                    break

    def _handle_email_search(self, query):
        results = self.search_emails(query)
        if not results:
            self.speak("No emails found.")
        else:
            self.speak(f"Found {len(results)} emails:")
            for idx, msg in results:
                self.speak(f"Email {idx}: {msg['subject']}")


# Run the Assistant
if __name__ == "__main__":
    assistant = VoiceEmailManager()
    assistant.run()
