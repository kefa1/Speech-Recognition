# main.py — FINAL FIXED VERSION (Intel Smart Sound + Channel 0 + Normalization)

import whisper
import sounddevice as sd
import numpy as np
import queue
import threading

print("Loading Whisper 'base' model...")
model = whisper.load_model("base")

q = queue.Queue()

def transcribe_worker():
    while True:
        audio = q.get()
        if audio is None:
            break

        print("   → Transcribing...")

        # FIX 1: Intel Smart Sound microphone → use channel 0 ONLY
        if audio.ndim > 1:
            audio = audio[:, 0]  # First channel carries actual speech

        # FIX 2: Normalize audio to [-1, 1]
        audio = audio.astype(np.float32)
        audio = audio / (np.max(np.abs(audio)) + 1e-9)

        # Optional debug:
        print("   Audio max amplitude:", np.max(np.abs(audio)))

        # Transcribe
        result = model.transcribe(audio, language="en", fp16=False)
        text = result["text"].strip()

        if text:
            print(f"   You said: {text}")
        else:
            print("   (no speech detected)")

        q.task_done()

# Start background transcriber
threading.Thread(target=transcribe_worker, daemon=True).start()

# Detect Intel Smart Sound WASAPI microphone
devices = sd.query_devices()
mic_id = None

for i, dev in enumerate(devices):
    if dev['max_input_channels'] > 0 and "Intel" in dev['name'] and "WASAPI" in sd.query_hostapis(dev['hostapi'])['name']:
        mic_id = i
        break

if mic_id is None:
    # fallback: any WASAPI microphone
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0 and "WASAPI" in sd.query_hostapis(dev['hostapi'])['name']:
            mic_id = i
            break

device = devices[mic_id]
print(f"Using: {device['name']}")
print(f"   Channels: {device['max_input_channels']}, Rate: {int(device['default_samplerate'])} Hz")

print("\nSpeak now! (5-second chunks | Ctrl+C to stop)")
print("-" * 60)

try:
    while True:
        print("\nRecording 5 seconds...")

        # KEY FIX: Exclusive mode + native 4 channels
        recording = sd.rec(
            frames=int(5 * device['default_samplerate']),
            samplerate=int(device['default_samplerate']),
            channels=device['max_input_channels'],
            device=mic_id,
            dtype='float32',
            blocking=True,
            extra_settings=sd.WasapiSettings(exclusive=True)
        )
        sd.wait()

        q.put(recording.copy())

except KeyboardInterrupt:
    print("\nStopping... Bye!")
    q.put(None)
