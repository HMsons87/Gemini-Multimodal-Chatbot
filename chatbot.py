import os
import sys
import customtkinter 
from google import genai 
from google.genai.types import Part
from tkinter import filedialog 
from PIL import Image         
import json 
import sounddevice as sd      
from scipy.io.wavfile import write 
import time 
import numpy as np 
from gtts import gTTS       
import pygame               
import threading            

# --- Constants ---
THEME_FILE = "theme_setting.txt"
DEFAULT_THEME = "System"
HISTORY_FILE = "chat_history.json" 

# --- Gemini API Setup ---
if "GEMINI_API_KEY" not in os.environ:
    print("ERROR: GEMINI_API_KEY environment variable is not set.")
    sys.exit(1)

client = genai.Client()
chat = None

# Initialize Pygame Mixer for Audio Playback
pygame.mixer.init()

# --- Helper Functions (Same as before) ---
def load_theme():
    try:
        with open(THEME_FILE, 'r') as f:
            theme = f.read().strip()
            if theme in ["Light", "Dark", "System"]:
                return theme
    except FileNotFoundError:
        pass
    return DEFAULT_THEME

def save_theme(theme_name):
    try:
        with open(THEME_FILE, 'w') as f:
            f.write(theme_name)
    except Exception as e:
        print(f"Error saving theme: {e}")

def load_history():
    try:
        with open(HISTORY_FILE, 'r') as f:
            data = json.load(f)
            return data.get('history', []) 
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_history(history_list):
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump({'history': history_list}, f, indent=4)
    except Exception as e:
        print(f"Error saving history: {e}")

# --- Chat Logic (Same as before) ---
def initialize_chat():
    global chat
    history = load_history()
    try:
        new_chat = client.chats.create(
            model="gemini-2.5-flash",
            history=history,
            config={
                'system_instruction': "You are a helpful AI assistant. Respond concisely. When receiving audio, listen carefully. If receiving an image, analyze it."
            }
        )
        return new_chat
    except Exception as e:
        print(f"Error initializing Gemini Chat: {e}")
        sys.exit(1)

chat = initialize_chat()

def get_chatbot_response(contents): 
    global chat 
    try:
        response = chat.send_message(contents) 
        save_history(chat.get_history()) 
        return response.text 
    except Exception as e:
        if "quota" in str(e).lower():
            return "Bot: ERROR: Quota exceeded. Check billing."
        return f"Bot: API Error: {e}"

# --- GUI Class ---
class ChatApp(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # Setup Theme
        last_theme = load_theme()
        customtkinter.set_appearance_mode(last_theme)
        
        # Setup Window
        self.title("Gemini AI Assistant - Ultimate Edition")
        self.geometry("600x800") 
        self.protocol("WM_DELETE_WINDOW", self.on_closing) 
        
        # Variables
        self.image_path = None
        self.audio_path = None
        self.is_recording = False
        self.audio_frames = []
        self.stream = None
        self.tts_enabled = customtkinter.BooleanVar(value=True) 
        
        # Layout: Chat History
        self.history_textbox = customtkinter.CTkTextbox(self, width=580, height=450, state="disabled")
        self.history_textbox.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="nsew")
        
        # Layout: Media Frame
        self.media_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        self.media_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        self.image_button = customtkinter.CTkButton(self.media_frame, text="Upload Image", command=self.select_image_file, width=100)
        self.image_button.pack(side="left", padx=5)
        
        self.image_label = customtkinter.CTkLabel(self.media_frame, text="No img", anchor="w", width=50)
        self.image_label.pack(side="left", padx=5)
        
        self.record_button = customtkinter.CTkButton(self.media_frame, text="Record Audio", command=self.toggle_recording, width=100)
        self.record_button.pack(side="right", padx=5)

        # TTS Switch
        self.tts_switch = customtkinter.CTkSwitch(self, text="Auto-Speak Response", variable=self.tts_enabled)
        self.tts_switch.grid(row=2, column=0, padx=10, pady=5, sticky="e")

        # Layout: Input
        self.input_entry = customtkinter.CTkEntry(self, width=450, placeholder_text="Type message...", font=customtkinter.CTkFont(size=14))
        self.input_entry.grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.input_entry.bind("<Return>", lambda event: self.send_message())

        self.send_button = customtkinter.CTkButton(self, text="Send", command=self.send_message, width=100)
        self.send_button.grid(row=3, column=0, padx=10, pady=5, sticky="e")
        
        # Layout: Controls
        self.controls_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        self.controls_frame.grid(row=4, column=0, padx=10, pady=10, sticky="ew")
        
        self.clear_button = customtkinter.CTkButton(self.controls_frame, text="Reset Chat", command=self.reset_chat, fg_color="#A00000", hover_color="#800000")
        self.clear_button.pack(side="left", padx=5)
        
        self.theme_switch_button = customtkinter.CTkButton(self.controls_frame, text="Theme", command=self.toggle_theme)
        self.theme_switch_button.pack(side="right", padx=5)
        
        # History and Welcome Message
        self.display_loaded_history()
        self.display_welcome_message() # NEW CALL HERE

    # --- TTS Functions ---
    def speak_text(self, text):
        """Converts text to speech in a separate thread."""
        if not self.tts_enabled.get() or not text:
            return

        def run_tts():
            try:
                # FIX: تنظيف النص من الرموز التي تشتت النطق
                clean_text = text.replace("*", "").replace("`", "").replace("[", "").replace("]", "").replace("_", " ").replace(":", ", ").strip()
                
                # يمكنك تغيير 'en' إلى 'ar' إذا كنت تريد الرد باللغة العربية
                language_code = 'en' 
                
                tts = gTTS(text=clean_text, lang=language_code) 
                temp_audio = "bot_response.mp3"
                tts.save(temp_audio)
                
                pygame.mixer.music.load(temp_audio)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                    
                pygame.mixer.music.unload()
            except Exception as e:
                print(f"TTS Error: {e}")

        threading.Thread(target=run_tts, daemon=True).start()

    # --- NEW: Welcome Message Function ---
    def display_welcome_message(self):
        # يمكنك تغيير النص إلى أي رسالة تريدها
        welcome_message = "Welcome! I am your AI assistant. I can see, hear, and talk. How can I help you today?"
        self.display_message("System", welcome_message)

    # --- Existing Methods (Recording, UI, etc.) ---
    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording_process()
        else:
            self.stop_recording_process()

    def start_recording_process(self):
        try:
            self.is_recording = True
            self.record_button.configure(text="Stop Rec", fg_color="red", hover_color="darkred")
            self.display_message("System", "Recording started...")
            
            self.audio_frames = [] 
            self.fs = 44100 

            def callback(indata, frames, time, status):
                if status: print(status, file=sys.stderr)
                self.audio_frames.append(indata.copy())

            self.stream = sd.InputStream(samplerate=self.fs, channels=1, callback=callback)
            self.stream.start()
            
        except Exception as e:
            self.display_message("System", f"Rec Error: {e}")
            self.is_recording = False
            self.record_button.configure(text="Record Audio", fg_color="#1F6AA5")

    def stop_recording_process(self):
        try:
            self.is_recording = False
            self.stream.stop()
            self.stream.close()
            
            recording = np.concatenate(self.audio_frames, axis=0)
            temp_file = f"temp_audio_{int(time.time())}.wav"
            write(temp_file, self.fs, (recording * 32767).astype(np.int16))
            
            self.audio_path = temp_file
            self.record_button.configure(text="Recorded", fg_color="green")
            self.display_message("System", "Audio saved.")
            
        except Exception as e:
            self.display_message("System", f"Save Error: {e}")
            self.reset_audio_selection()

    def reset_audio_selection(self):
        if self.audio_path and os.path.exists(self.audio_path):
             try: os.remove(self.audio_path)
             except: pass
        self.audio_path = None
        self.is_recording = False 
        self.record_button.configure(text="Record Audio", fg_color="#1F6AA5", state="normal")

    def select_image_file(self):
        file_path = filedialog.askopenfilename(filetypes=(("Images", "*.png;*.jpg;*.jpeg"), ("All", "*.*")))
        if file_path:
            self.image_path = file_path
            self.image_label.configure(text=f"Img selected")
        else:
            self.reset_image_selection()

    def reset_image_selection(self):
        self.image_path = None
        self.image_label.configure(text="No img")

    def display_loaded_history(self):
        global chat
        history = chat.get_history()
        self.display_message("System", f"Ready.")
        if history:
            for msg in history:
                role = "You" if msg.role == "user" else "Bot"
                try: text = msg.parts[0].text or "[Media]"
                except: text = "[Media]"
                self.display_message(role, text)

    def on_closing(self):
        save_history(chat.get_history()) 
        if self.stream: self.stream.close()
        if os.path.exists("bot_response.mp3"):
            try: os.remove("bot_response.mp3")
            except: pass
        self.destroy() 

    def toggle_theme(self):
        new = "Light" if customtkinter.get_appearance_mode() == "Dark" else "Dark"
        customtkinter.set_appearance_mode(new)
        save_theme(new)

    def reset_chat(self):
        global chat
        chat = initialize_chat()
        self.history_textbox.configure(state="normal")
        self.history_textbox.delete("1.0", "end")
        self.history_textbox.configure(state="disabled")
        if os.path.exists(HISTORY_FILE):
             try: os.remove(HISTORY_FILE)
             except: pass
        self.reset_image_selection()
        self.reset_audio_selection()
        self.display_message("System", "Reset done.")

    def display_message(self, sender, message):
        self.history_textbox.configure(state="normal")
        self.history_textbox.insert("end", f"[{sender}]: {message}\n\n")
        self.history_textbox.see("end")  
        self.history_textbox.configure(state="disabled")
        
        if sender == "Bot" and "Error" not in message:
            self.speak_text(message)

    def send_message(self):
        user_prompt = self.input_entry.get()
        image_path = self.image_path
        audio_path = self.audio_path
        
        if not user_prompt.strip() and not image_path and not audio_path:
            return 

        contents = []
        display_text = user_prompt
        
        if audio_path:
            try:
                uploaded_file = client.files.upload(file=audio_path, config={'mime_type': 'audio/wav'})
                contents.insert(0, uploaded_file)
                display_text += f" [Audio Sent]"
            except Exception as e:
                self.display_message("System", f"Audio Error: {e}")
                self.reset_audio_selection()
                return

        if image_path:
            try:
                img = Image.open(image_path)
                contents.insert(0, img)
                display_text += f" [Image Sent]"
            except Exception as e:
                self.display_message("System", f"Image Error: {e}")
                self.reset_image_selection()
                return
        
        if user_prompt.strip():
            contents.append(user_prompt)
        elif audio_path and not image_path:
             contents.append("Listen to the audio and respond.")
        
        if not contents: return

        self.display_message("You", display_text)
        self.input_entry.delete(0, "end")
        self.reset_image_selection()
        self.reset_audio_selection() 
        
        threading.Thread(target=self.run_api_call, args=(contents,), daemon=True).start()

    def run_api_call(self, contents):
        try:
            bot_response = get_chatbot_response(contents) 
        except Exception as e:
             bot_response = f"Bot Error: {e}"
        
        self.after(0, lambda: self.display_message("Bot", bot_response))

if __name__ == "__main__":
    app = ChatApp()
    app.mainloop()