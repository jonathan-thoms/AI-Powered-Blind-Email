import imaplib
import email
import re
import speech_recognition as sr
import pyttsx3
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from transformers import pipeline
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration


# Configure Hugging Face cache
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
cache_dir = os.path.join(os.getcwd(), "huggingface_cache")
os.makedirs(cache_dir, exist_ok=True)


class VoiceEmailManager:
    def __init__(self):
        # Email Credentials
        self.EMAIL_USER = "emailtest.jonathan@gmail.com"
        self.EMAIL_PASS = "xwis udwz bwgr ueuh"
        # Initialize summarizer (add this inside __init__)
        self.summarizer = pipeline(
            "summarization",
            model = "sshleifer/distilbart-cnn-12-6",  # Powerful but slower
            device=-1,  # Use CPU
            min_length=30,
            max_length=100
        )
        # Voice Engine
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 150)
        self.recognizer = sr.Recognizer()

        # Spam Filter Model (Naive Bayes)
        self.vectorizer = TfidfVectorizer()
        self.spam_model = MultinomialNB()
        self._train_dummy_spam_model()

        # Initialize image processor with fast tokenizer
        self.image_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            use_fast=True  # Force fast tokenizer
        )
        self.image_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )


        # Priority Detection Rules
        self.priority_senders = ["important@domain.com", "boss@company.com"]
        self.priority_keywords = ["urgent", "meeting", "deadline"]

        # Current state
        self.current_emails = {}

    # --- Core Functions ---
    def speak(self, text):
        print(f"ASSISTANT: {text}")
        self.engine.say(text)
        self.engine.runAndWait()

    def listen_command(self, max_retries=3):
        for attempt in range(max_retries):
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                self.speak("Listening..." if attempt == 0 else "Try again...")
                try:
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=8)
                    command = self.recognizer.recognize_google(audio).lower()
                    print(f"USER SAID: {command}")
                    return command
                except sr.WaitTimeoutError:
                    self.speak("I didn't hear anything.")
                except sr.UnknownValueError:
                    self.speak("Sorry, I didn't understand that.")
                except sr.RequestError:
                    self.speak("Speech service error.")
        return input("Type your command: ").lower()

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
            print(f"ERROR: {str(e)}")
            return {}

    # --- Email Processing ---
    def _describe_image(self, image_data):
        """Robust image description with error handling"""
        try:
            img = Image.open(BytesIO(image_data))

            # Convert to RGB if needed (for PNGs with transparency)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            inputs = self.image_processor(img, return_tensors="pt")
            out = self.image_model.generate(**inputs, max_new_tokens=50)
            return self.image_processor.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Image processing failed: {e}")
            return None

    def _get_email_body(self, msg, include_image_descriptions=True):  # Changed default to True
        """
        Enhanced version that now automatically describes images by default
        while maintaining backward compatibility
        """
        text_body = ""
        image_descriptions = []

        try:
            if msg.is_multipart():
                for part in msg.walk():
                    # Get text content
                    if part.get_content_type() == "text/plain":
                        text_body = part.get_payload(decode=True).decode("utf-8", errors="ignore")

                    # Auto-detect images when enabled
                    if include_image_descriptions and part.get_content_maintype() == 'image':
                        img_data = part.get_payload(decode=True)
                        description = self._describe_image(img_data)
                        if description:
                            image_descriptions.append(description)

            else:  # Non-multipart email
                text_body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")

        except Exception as e:
            print(f"Email parsing error: {e}")
            text_body = "Could not read email body"

        # Return format depends on whether image descriptions are requested
        if include_image_descriptions:
            return {
                'text': text_body,
                'images': image_descriptions if image_descriptions else None
            }
        return text_body  # Backward compatible return

    def _get_email_images(self, msg):
        """Extract all images from email"""
        images = []
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_maintype() == 'image':
                    try:
                        img_data = part.get_payload(decode=True)
                        images.append(img_data)
                    except Exception as e:
                        print(f"Failed to extract image: {e}")
        return images

    def read_email_with_images(self, email_list, selected_index):
        """Enhanced email reader with image descriptions"""
        if selected_index not in email_list:
            self.speak("Invalid selection")
            return

        _, msg = email_list[selected_index]

        # Get text body (original functionality)
        text_body = self._get_email_body(msg)

        # Process images
        image_descriptions = []
        for img_data in self._get_email_images(msg):
            desc = self._describe_image(img_data)
            if desc:
                image_descriptions.append(desc)

        # Build output
        if image_descriptions:
            self.speak("Found images:")
            for i, desc in enumerate(image_descriptions, 1):
                self.speak(f"Image {i}: {desc}")

        self.speak("Email content: " + text_body[:300])

    def extract_number(self, command):
        """Improved number extraction from voice commands"""
        # Check for digits ("open 3")
        num_match = re.search(r'\b(\d+)\b', command)
        if num_match:
            return int(num_match.group(1))

        # Check for number words ("open three")
        number_words = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'first': 1, 'second': 2, 'third': 3, 'last': len(self.current_emails)
        }
        for word, num in number_words.items():
            if word in command.lower():
                return num
        return None

    # --- Features ---
    def _train_dummy_spam_model(self):
        X_train = ["win free money", "meeting at 3pm", "urgent update"]
        y_train = [1, 0, 0]  # 1=spam
        X_vec = self.vectorizer.fit_transform(X_train)
        self.spam_model.fit(X_vec, y_train)

    def is_spam(self, email_text):
        try:
            X_vec = self.vectorizer.transform([email_text])
            return self.spam_model.predict(X_vec)[0] == 1
        except:
            return False

    def is_priority(self, email_msg):
        subject = email_msg.get("subject", "").lower()
        sender = email_msg.get("from", "").lower()

        # Get just the text body (no images)
        body_data = self._get_email_body(email_msg, include_image_descriptions=False)
        body = body_data.lower() if isinstance(body_data, str) else ""

        return (any(s.lower() in sender for s in self.priority_senders) or
                any(k.lower() in subject or k.lower() in body for k in self.priority_keywords))

    def _chunk_text(self, text, chunk_size=1024):
        """Split long emails into manageable chunks"""
        words = text.split()
        for i in range(0, len(words), chunk_size):
            yield ' '.join(words[i:i + chunk_size])

    def summarize_email(self, email_msg):
        """Analyzes entire email and provides concise summary"""
        full_body = self._get_email_body(email_msg)

        # Skip very short emails
        if len(full_body.split()) < 25:
            return "Brief email: " + full_body[:150] + ("..." if len(full_body) > 150 else "")

        try:
            # Process in chunks if needed (for long emails)
            if len(full_body) > 1024:
                chunks = list(self._chunk_text(full_body))
                summaries = []
                for chunk in chunks:
                    summary = self.summarizer(chunk, max_length=75, min_length=25)[0]["summary_text"]
                    summaries.append(summary)
                full_summary = ' '.join(summaries)
                # Summarize the summaries if too long
                if len(full_summary.split()) > 50:
                    full_summary = self.summarizer(full_summary, max_length=100, min_length=30)[0]["summary_text"]
            else:
                full_summary = self.summarizer(full_body)[0]["summary_text"]

            # Clean up and ensure proper sentence structure
            full_summary = full_summary.replace(" .", ".").replace(" ,", ",")
            full_summary = re.sub(r'\s+', ' ', full_summary).strip()
            return full_summary

        except Exception as e:
            print(f"Summarization error: {e}")
            # Fallback to key sentences
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', full_body) if s.strip()]
            if len(sentences) >= 3:
                return f"Main points: {sentences[0]} {sentences[len(sentences) // 2]} {sentences[-1]}"
            return sentences[0] if sentences else "Could not generate summary"

    # --- Main Workflows ---
    def _handle_email_checking(self):
        self.current_emails = self.fetch_emails()
        if not self.current_emails:
            self.speak("No new emails found.")
            return

        # List emails with priority/spam flags
        for idx, (email_id, msg) in self.current_emails.items():
            flags = []
            if self.is_priority(msg):
                flags.append("Priority")

            # Use text-only version for spam detection
            text_only_body = self._get_email_body(msg, include_image_descriptions=False)
            if self.is_spam(f"{msg['subject']} {text_only_body}"):
                flags.append("Spam")

            status = f" ({' | '.join(flags)})" if flags else ""
            self.speak(f"Email {idx}: From {msg['from']}. Subject: {msg['subject']}{status}")

        self.speak("Say 'open X' to read, 'describe X' for images only, or 'summarize X' for summary.")

        while True:
            cmd = self.listen_command()
            if not cmd:
                continue

            if "back" in cmd.lower():
                break

            num = self.extract_number(cmd)
            if num in self.current_emails:
                msg = self.current_emails[num][1]

                if "summarise" in cmd.lower() or "summarize" in cmd.lower():
                    # Use text-only version for summarization
                    text_body = self._get_email_body(msg, include_image_descriptions=False)
                    summary = self.summarize_email(msg)
                    self.speak(f"Summary: {summary}")

                elif "describe" in cmd.lower():
                    # Image-only mode
                    result = self._get_email_body(msg, include_image_descriptions=True)
                    if result.get('images'):
                        self.speak("Found images:")
                        for i, desc in enumerate(result['images'], 1):
                            self.speak(f"Image {i}: {desc}")
                    else:
                        self.speak("No images found in this email.")

                else:
                    # Default reading mode
                    result = self._get_email_body(msg, include_image_descriptions=True)
                    if result.get('images'):
                        self.speak("This email contains images:")
                        for i, desc in enumerate(result['images'], 1):
                            self.speak(f"Image {i}: {desc}")

                    self.speak("Email content: " + result['text'][:300])
            else:
                self.speak("Invalid selection. Try again.")

    def _describe_email_images(self, msg):
        """Describe all images in an email"""
        image_descriptions = []
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_maintype() == 'image':
                    try:
                        img_data = part.get_payload(decode=True)
                        description = self._describe_image(img_data)
                        if description:
                            image_descriptions.append(description)
                    except Exception as e:
                        print(f"Failed to process image: {e}")

        if image_descriptions:
            self.speak(f"Found {len(image_descriptions)} images:")
            for i, desc in enumerate(image_descriptions, 1):
                self.speak(f"Image {i}: {desc}")
        else:
            self.speak("No images found in this email.")

    def _handle_email_search(self, initial_query=None):
        """Fixed search workflow"""
        query = initial_query
        if not query:
            self.speak("What would you like to search for?")
            query = self.listen_command()
            if not query:
                self.speak("Search cancelled.")
                return

        results = self.search_emails(query)
        if not results:
            self.speak(f"No emails found matching '{query}'.")
            return

        self.speak(f"Found {len(results)} results:")
        for idx, msg in results:
            self.speak(f"Email {idx}: {msg['subject']}")

        # Stay in search context
        while True:
            self.speak("Say 'open X' to read, or 'back' to return.")
            cmd = self.listen_command()

            if not cmd:
                continue
            if "back" in cmd.lower():
                break

            num = self.extract_number(cmd)
            if num in dict(results):
                self.speak(self._get_email_body(dict(results)[num][1]))
            else:
                self.speak("Invalid selection. Try again.")

    def run(self):
        self.speak("Welcome to your voice-controlled email assistant.")
        while True:
            self.speak("Say 'check email', 'search', or 'exit'.")
            command = self.listen_command()

            if not command:
                continue
            elif "check" in command or "email" in command:
                self._handle_email_checking()
            elif "search" in command:
                self._handle_email_search(command.replace("search", "").strip())
            elif "exit" in command or "quit" in command:
                self.speak("Goodbye!")
                break


if __name__ == "__main__":
    assistant = VoiceEmailManager()
    assistant.run()
