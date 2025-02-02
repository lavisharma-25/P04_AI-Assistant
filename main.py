import os
import whisper
import pyttsx3
import pyaudio
import wave
from dotenv import load_dotenv
import google.generativeai as genai
import warnings

# Suppress warnings from Whisper
warnings.filterwarnings("ignore", category=UserWarning, module="whisper")

# Load environment variables
load_dotenv()

# Add ffmpeg to PATH
os.environ["PATH"] += os.pathsep + r"ffmpeg\bin"

# Initialize Whisper Model for Speech-to-Text
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")

# Initialize TTS engine (pyttsx3)
tts_engine = pyttsx3.init()

# Initialize the Gemini 1.5 Flash model
print("Loading Gemini 1.5 Flash model for text generation...")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Audio settings for pyaudio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper works best with 16 kHz audio
CHUNK = 1024


# Function to record live audio
def record_audio(seconds=5):
    """
    Records live audio from the microphone and saves it as a WAV file.

    Args:
        seconds (int): Duration of the recording in seconds.

    Returns:
        str: Path to the saved audio file or None if an error occurs.
    """
    try:
        print("Recording... Speak now!")
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        frames = []

        for _ in range(0, int(RATE / CHUNK * seconds)):
            data = stream.read(CHUNK)
            frames.append(data)

        print("Recording complete!")
        stream.stop_stream()
        stream.close()
        audio.terminate()

        # Save recorded audio to a file
        current_script_path = os.path.dirname(os.path.abspath(__file__))
        audio_file = os.path.join(current_script_path, "Audio_Data", "live_audio.wav")
        os.makedirs(os.path.dirname(audio_file), exist_ok=True)
        wf = wave.open(audio_file, "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        return audio_file
    except Exception as e:
        print(f"Error during audio recording: {e}")
        return None


# Function to transcribe audio (speech-to-text)
def transcribe_audio(audio_file):
    """
    Transcribes audio from the given file using Whisper.

    Args:
        audio_file (str): Path to the audio file.

    Returns:
        str: Transcribed text or None if an error occurs.
    """
    try:
        print("Transcribing audio...")
        result = whisper_model.transcribe(audio_file)
        text = result["text"]
        print(f"User said: {text}")
        return text
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None


# Function to generate model response using Gemini 1.5 Flash
def get_gpt_response(text_input):
    """
    Generates a response to the given text input using Gemini 1.5 Flash.

    Args:
        text_input (str): The text input to process.

    Returns:
        str: Generated response text or None if an error occurs.
    """
    try:
        print("Processing input with Gemini 1.5 Flash...")
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            },
            system_instruction="""You are a chat assistant expert in human conversation.  Your responses should:
                * Be natural and engaging, avoiding overly formal or technical language.
                * Show empathy and understanding of the user's perspective.
                * Adapt to the user's communication style and emotional tone.
                * Provide helpful and relevant information while maintaining a conversational flow.
                * Ask clarifying questions when necessary to ensure understanding.
                * Avoid making assumptions or providing inaccurate information."""

        )
        response = model.generate_content(text_input)
        return response.text
    except Exception as e:
        print(f"Error during text generation: {e}")
        return None


# Function to convert text to speech and play it
def text_to_speech(text):
    """
    Converts the given text to speech and plays it.

    Args:
        text (str): The text to convert to speech.
    """
    try:
        print(f"AI: {text}")
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        print(f"Error during text-to-speech: {e}")


# Main Function
def main():
    """
    Main function to run the AI assistant.
    """
    try:
        greeting_message = "Hello! How can I assist you today?"
        print(greeting_message)
        text_to_speech(greeting_message)

        while True:
            print("\n--- Starting Chatbot ---")

            # Step 1: Record audio
            audio_file = record_audio(seconds=5)
            if not audio_file:
                print("Error during recording. Please try again.")
                continue

            # Step 2: Transcribe audio
            text_input = transcribe_audio(audio_file)
            if not text_input:
                print("Unable to transcribe audio. Please try again.")
                continue

            # Step 3: Check if the user wants to quit
            if "quit" in text_input.lower() or "exit" in text_input.lower():
                goodbye_message = "Goodbye, have a great day!"
                print(goodbye_message)
                text_to_speech(goodbye_message)
                break

            # Step 4: Generate AI response
            response = get_gpt_response(text_input)
            if not response:
                print("Error generating response. Please try again.")
                continue

            # Step 5: Convert response to speech
            text_to_speech(response)
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    finally:
        tts_engine.stop()


if __name__ == "__main__":
    main()
