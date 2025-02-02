import whisper
import time
import librosa
import soundfile as sf
import torch
import pandas as pd
from jiwer import wer, cer, Compose, RemovePunctuation, ToLowerCase, Strip
from difflib import SequenceMatcher

# Convert MP3 to WAV
input_mp3_path = r"C:\Users\Bloodycub\Desktop\RU_F_OlgaT.mp3"  # Update path if needed
wav_path = "RU_F_OlgaT.wav"

# Load and convert audio to WAV format
audio, sr = librosa.load(input_mp3_path, sr=16000)  # Convert to 16kHz (Whisper optimal)
sf.write(wav_path, audio, sr)

# Define Whisper models to benchmark
models = ["tiny", "small", "medium", "large","turbo", "large-v3", "large-v3-turbo"]

# Ground truth transcription (Modify with correct Russian text)
ground_truth_text = """я бываю такой разной и только здесь знает какая я настоящая
следуй за своим сердцем следуй в гурмэ деликатесы в атмосфере люкса и вместе лучшие коллекции
швейцарских часов dolce vita от бензони аромат шоколада окутывает вашу душу
вызывая страстную зависимость от божественных лакомств"""

# Jiwer Transformation Pipeline to Ensure Proper Normalization
transform = Compose([
    RemovePunctuation(),  # Remove punctuation
    ToLowerCase(),        # Convert to lowercase
    Strip()               # Remove extra spaces
])

# Function to calculate accuracy
def calculate_accuracy(reference, hypothesis):
    reference = transform(reference)  # Normalize ground truth
    hypothesis = transform(hypothesis)  # Normalize transcribed text

    # Debug: Print preprocessed texts for verification
    print(f"\n🔹 Ground Truth (Processed):\n{reference}")
    print(f"🔹 Transcription (Processed):\n{hypothesis}")

    word_error = wer(reference, hypothesis)  # Word Error Rate (WER)
    char_error = cer(reference, hypothesis)  # Character Error Rate (CER)
    similarity = SequenceMatcher(None, reference, hypothesis).ratio() * 100  # Text Similarity %

    word_accuracy = max(0, round((1 - word_error) * 100, 2))  # Prevent negative values
    char_accuracy = max(0, round((1 - char_error) * 100, 2))  # Prevent negative values

    print(f"✅ Word Accuracy: {word_accuracy}%")
    print(f"✅ Character Accuracy: {char_accuracy}%")
    print(f"✅ Text Similarity: {similarity}%")

    return word_accuracy, char_accuracy, round(similarity, 2)

# Run benchmark
results = []

# Check if GPU is available and force GPU testing
if torch.cuda.is_available():
    devices = ["cuda", "cpu"]  # Run on both GPU and CPU
else:
    devices = ["cpu"]  # If no GPU, only run on CPU

print("\n🚀 Detected Devices:", devices)

# Test each model on GPU (if available) and CPU
for device in devices:
    print(f"\n🚀 Running Whisper on: {device.upper()}")

    for model_name in models:
        print(f"\n🚀 Testing model: {model_name} on {device.upper()}...")

        # Load model on specified device
        model = whisper.load_model(model_name, device=device)

        # Transcribe audio
        start_time = time.time()
        result = model.transcribe(wav_path, language="ru")
        end_time = time.time()

        # Get transcription and time taken
        transcribed_text_original = result["text"].strip().lower()  # Convert transcription to lowercase
        transcribed_text_clean = transform(transcribed_text_original)  # Normalize transcription

        time_taken = round(end_time - start_time, 2)

        # Compute accuracy
        word_acc, char_acc, similarity = calculate_accuracy(ground_truth_text, transcribed_text_clean)

        # Ensure accuracy values are being stored correctly
        model_result = {
            "Model": model_name,
            "Device": device.upper(),
            "Time (s)": time_taken,
            "Word Accuracy (%)": word_acc,
            "Character Accuracy (%)": char_acc,
            "Text Similarity (%)": similarity,
            "Transcribed Text (No Punctuation & Lowercase)": transcribed_text_clean
        }

        results.append(model_result)

        # Print results immediately after testing each model
        print("\n📊 Benchmark Results for Model:", model_name)
        for key, value in model_result.items():
            print(f"{key}: {value}")
        print("-" * 80)  # Separator for better readability

# Convert results to DataFrame **WITH PROPER COLUMN ORDER**
df_results = pd.DataFrame(results, columns=[
    "Model", "Device", "Time (s)", "Word Accuracy (%)", 
    "Character Accuracy (%)", "Text Similarity (%)", 
    "Transcribed Text (No Punctuation & Lowercase)"
])

# Display final results
print("\n📊 Final Benchmark Results:")
print(df_results)

# Save results to CSV
df_results.to_csv("whisper_benchmark_results.csv", index=False)
