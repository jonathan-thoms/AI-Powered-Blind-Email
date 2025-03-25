import imaplib
import email
import speech_recognition as sr
import pyttsx3
import re  # Import regex for better number detection

# Email Credentials
EMAIL_USER = "emailtest.jonathan@gmail.com"
EMAIL_PASS = "xwis udwz bwgr ueuh"

# Text-to-Speech Engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # Adjust speed


# Function to Speak Text
def speak(text):
    engine.say(text)
    engine.runAndWait()


# Function to Recognize Speech
def listen_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        speak("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            command = recognizer.recognize_google(audio).lower()
            return command
        except sr.UnknownValueError:
            speak("Sorry, I couldn't understand. Try again.")
        except sr.RequestError:
            speak("There was an issue with the speech recognition service.")
    return None


# Function to Fetch Unread Emails (Sender + Subject Only)
def fetch_unread_emails():
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(EMAIL_USER, EMAIL_PASS)
        mail.select("inbox")

        _, messages = mail.search(None, "UNSEEN")  # Fetch unread emails
        email_ids = messages[0].split()

        if not email_ids:
            speak("You have no unread emails.")
            return {}

        email_list = {}  # Dictionary to store email details

        speak(f"You have {len(email_ids)} unread emails.")

        for idx, email_id in enumerate(email_ids[:5], start=1):  # Fetch up to 5 emails
            _, msg_data = mail.fetch(email_id, "(RFC822)")
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    subject = msg["subject"]
                    sender = msg["from"]

                    email_list[idx] = (email_id, msg)  # Store email details

                    speak(f"Email {idx}: From {sender}. Subject: {subject}")

        mail.logout()
        return email_list  # Return the dictionary of emails
    except Exception as e:
        speak("Error fetching emails.")
        print(str(e))
        return {}


# Function to Read Selected Email Body
def read_email_body(email_list, selected_index):
    if selected_index in email_list:
        _, msg = email_list[selected_index]

        speak("Reading email body.")
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode("utf-8")
                    speak(body[:300])  # Read first 300 chars
                    break
        else:
            body = msg.get_payload(decode=True).decode("utf-8")
            speak(body[:300])

    else:
        speak("Invalid email selection.")


# Function to Extract Email Number from Speech
# Mapping of number words to digits
number_words = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}


def extract_email_number(command):
    # Try to find a digit-based number
    match = re.search(r"(\d+)", command)
    if match:
        return int(match.group(1))  # Convert extracted number to integer

    # Check for number words (e.g., "open email one")
    words = command.split()
    for word in words:
        if word in number_words:
            return number_words[word]  # Convert word to integer

    return None  # Return None if no number is found


# Main Program
speak("Welcome to your voice-controlled email assistant.")

while True:
    speak("Say 'check email' to hear your inbox, or 'exit' to quit.")
    command = listen_command()

    if command == "check email":
        emails = fetch_unread_emails()

        if emails:
            speak("Say 'Open Email' followed by the number to read its content.")
            while True:
                selection_command = listen_command()
                if selection_command:
                    selected_number = extract_email_number(selection_command)
                    if selected_number > 0:
                        read_email_body(emails, selected_number)
                        break
                    else:
                        speak("Invalid selection. Please say 'Open Email' followed by a number.")
    elif command == "exit":
        speak("Goodbye!")
        break
    else:
        speak("Invalid command. Try again.")
