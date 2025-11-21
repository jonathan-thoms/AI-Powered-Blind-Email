import os
import sys
import imaplib
import email
import re
import speech_recognition as sr
import pyttsx3
import torch
from typing import Optional, List, Dict, Any, Generator
from io import BytesIO
from dotenv import load_dotenv
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration

# Load environment variables
load_dotenv()

# Constants
HF_CACHE_DIR = os.path.join(os.getcwd(), "huggingface_cache")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.makedirs(HF_CACHE_DIR, exist_ok=True)

class VoiceEmailManager:
    """
    Core logic for the Voice Email Assistant. 
    Handles IMAP connections, Speech-to-Text, Text-to-Speech, 
    AI Summarization, and Image Captioning.
    """

    def __init__(self):
        self.email_user = os.getenv("EMAIL_USER")
        self.email_pass = os.getenv("EMAIL_PASS")
        
        if not self.email_user or not self.email_pass:
            print("ERROR: Credentials not found. Please check your .env file.")

        # Initialize Text-to-Speech
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 150)
        
        # Initialize Speech-to-Text
        self.recognizer = sr.Recognizer()

        # Initialize AI Models (Lazy loading could be better, but init is fine for now)
        self._init_ai_models()
        self._train_spam_filter()

        # Configuration
        self.priority_senders = ["important@domain.com", "boss@company.com"]
        self.priority_keywords = ["urgent", "meeting", "deadline"]
        self.current_emails: Dict[int, Any] = {}

    def _init_ai_models(self):
        """Initialize HuggingFace pipelines and models."""
        print("Loading AI Models...")
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=0 if torch.cuda.is_available() else -1
        )
        
        self.image_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base", use_fast=True
        )
        self.image_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

    def _train_spam_filter(self):
        """Trains a lightweight Naive Bayes spam classifier."""
        self.vectorizer = TfidfVectorizer()
        self.spam_model = MultinomialNB()
        
        # Dummy training data
        X_train = ["win free money", "meeting at 3pm", "urgent update", "lottery winner"]
        y_train = [1, 0, 0, 1]  # 1=spam
        
        X_vec = self.vectorizer.fit_transform(X_train)
        self.spam_model.fit(X_vec, y_train)

    # --- Interaction Methods ---

    def speak(self, text: str):
        """Outputs text to speech and console."""
        print(f"ASSISTANT: {text}")
        self.engine.say(text)
        self.engine.runAndWait()

    def listen_command(self, max_retries: int = 3) -> str:
        """Captures audio input and converts to text."""
        for attempt in range(max_retries):
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                self.speak("Listening..." if attempt == 0 else "Try again...")
                try:
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=8)
                    command = self.recognizer.recognize_google(audio).lower()
                    print(f"USER SAID: {command}")
                    return command
                except (sr.WaitTimeoutError, sr.UnknownValueError):
                    continue
                except sr.RequestError:
                    self.speak("Speech service unavailable.")
                    break
        
        # Fallback for testing/noisy environments
        return input("Type your command (Fallback): ").lower()

    # --- Email Logic ---

    def fetch_emails(self, folder: str = "inbox", limit: int = 10, unseen: bool = True) -> Dict[int, Any]:
        """Fetches emails from IMAP server."""
        try:
            mail = imaplib.IMAP4_SSL("imap.gmail.com")
            mail.login(self.email_user, self.email_pass)
            mail.select(folder)

            search_crit = "UNSEEN" if unseen else "ALL"
            status, messages = mail.search(None, search_crit)
            email_ids = messages[0].split()

            # Get latest emails first
            email_ids = email_ids[::-1][:limit]

            email_list = {}
            for idx, email_id in enumerate(email_ids, start=1):
                _, msg_data = mail.fetch(email_id, "(RFC822)")
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        email_list[idx] = (email_id, msg)
            
            mail.logout()
            return email_list
        except Exception as e:
            self.speak("Connection error.")
            print(f"IMAP ERROR: {e}")
            return {}

    def get_email_content(self, msg, include_images: bool = True) -> Dict[str, Any]:
        """Parses email for text body and generates image captions."""
        text_body = ""
        captions = []

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))

                if content_type == "text/plain" and "attachment" not in content_disposition:
                    text_body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                
                if include_images and part.get_content_maintype() == 'image':
                    try:
                        img_data = part.get_payload(decode=True)
                        caption = self._generate_image_caption(img_data)
                        if caption:
                            captions.append(caption)
                    except Exception:
                        pass
        else:
            text_body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")

        return {'text': text_body, 'images': captions}

    def _generate_image_caption(self, image_data: bytes) -> Optional[str]:
        """Uses BLIP model to caption an image."""
        try:
            img = Image.open(BytesIO(image_data)).convert('RGB')
            inputs = self.image_processor(img, return_tensors="pt")
            out = self.image_model.generate(**inputs, max_new_tokens=50)
            return self.image_processor.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Image Caption Error: {e}")
            return None

    def summarize_text(self, text: str) -> str:
        """Summarizes long email bodies using BART."""
        clean_text = re.sub(r'\s+', ' ', text).strip()
        if len(clean_text.split()) < 30:
            return clean_text

        try:
            # Chunk text if too long for model
            max_chunk = 1024
            chunks = [clean_text[i:i+max_chunk] for i in range(0, len(clean_text), max_chunk)]
            
            summaries = []
            for chunk in chunks:
                summary = self.summarizer(chunk, max_length=100, min_length=30, truncation=True)
                summaries.append(summary[0]['summary_text'])
            
            return " ".join(summaries)
        except Exception as e:
            print(f"Summarization failed: {e}")
            return "Could not summarize email."

    # --- Workflows ---

    def handle_check_emails(self):
        self.current_emails = self.fetch_emails(unseen=True)
        if not self.current_emails:
            self.speak("You have no new unread emails.")
            return

        self.speak(f"You have {len(self.current_emails)} new emails.")
        
        for idx, (_, msg) in self.current_emails.items():
            sender = msg.get("from", "Unknown")
            subject = msg.get("subject", "No Subject")
            self.speak(f"Email {idx}: From {sender}. Subject: {subject}")

        self.speak("Say 'read one', 'summarize two', or 'describe images in three'.")
        self._email_interaction_loop()

    def _email_interaction_loop(self):
        while True:
            cmd = self.listen_command()
            if not cmd or "back" in cmd or "exit" in cmd:
                break

            num = self._extract_number(cmd)
            if num and num in self.current_emails:
                msg = self.current_emails[num][1]
                content = self.get_email_content(msg)

                if "summar" in cmd:
                    summary = self.summarize_text(content['text'])
                    self.speak(f"Summary: {summary}")
                elif "describe" in cmd or "image" in cmd:
                    if content['images']:
                        for i, cap in enumerate(content['images'], 1):
                            self.speak(f"Image {i}: {cap}")
                    else:
                        self.speak("No images found.")
                else:
                    self.speak(f"Body: {content['text'][:500]}")
                    if content['images']:
                        self.speak("This email also contains images. Say 'describe images' to hear them.")
            else:
                self.speak("Please select a valid email number.")

    def _extract_number(self, text: str) -> Optional[int]:
        """Extracts integer from voice command."""
        word_map = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5}
        for word, digit in word_map.items():
            if word in text: return digit
        
        match = re.search(r'\b(\d+)\b', text)
        return int(match.group(1)) if match else None

    def run(self):
        self.speak("Voice Email Assistant Online.")
        while True:
            self.speak("Commands: Check Email, Search, or Exit.")
            cmd = self.listen_command()
            
            if "check" in cmd:
                self.handle_check_emails()
            elif "exit" in cmd or "quit" in cmd:
                self.speak("Goodbye.")
                break

if __name__ == "__main__":
    assistant = VoiceEmailManager()
    assistant.run()
