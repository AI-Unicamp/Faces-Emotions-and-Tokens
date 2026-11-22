import argparse
from furhat_realtime_api import FurhatClient
import logging
import time
import threading
import winsound
import tempfile
import os
from pydub import AudioSegment


parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="127.0.0.1", help="Furhat robot IP address")
parser.add_argument("--auth_key", type=str, default=None, help="Authentication key for Realtime API")
parser.add_argument("--src", type=str, default="data/blendshapes_converted_test1", help="Path to .txt file (without extension)")
parser.add_argument("--wav", type=str, default=None, help="Path to .wav file (if different from src)")
parser.add_argument("--start_sec", type=float, default=0.0, help="Audio start time in seconds")
parser.add_argument("--duration", type=float, default=None, help="Audio duration in seconds (None = play all)")
args = parser.parse_args()

def convert_float32_to_pcm16(path):
    """Convert float32 WAV to pcm_s16le WAV using only numpy + struct"""
    import struct
    import numpy as np

    with open(path, 'rb') as f:
        # Read RIFF header
        riff = f.read(4)
        if riff != b'RIFF':
            raise ValueError("Not a WAV file")
        f.read(4)  # file size
        f.read(4)  # WAVE

        # Find fmt and data chunks
        fmt_data = None
        audio_data = None
        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size = struct.unpack('<I', f.read(4))[0]
            if chunk_id == b'fmt ':
                fmt_data = f.read(chunk_size)
            elif chunk_id == b'data':
                audio_data = f.read(chunk_size)
            else:
                f.read(chunk_size)

    fmt_tag, n_channels, sample_rate = struct.unpack('<HHI', fmt_data[:8])

    # Read float32 samples
    samples = np.frombuffer(audio_data, dtype=np.float32)

    # Convert to int16
    samples = np.clip(samples, -1.0, 1.0)
    samples_int16 = (samples * 32767).astype(np.int16)

    # Write as pcm_s16le WAV
    out_path = path.replace(".wav", "_pcm16.wav")
    import wave
    with wave.open(out_path, 'wb') as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(samples_int16.tobytes())

    print(f"Converted: {out_path} ({sample_rate}Hz, {n_channels}ch, s16)")
    return out_path

'''def play_audio_segment(path, start_sec=0.0, duration_sec=None):
    from pydub import AudioSegment

    audio = AudioSegment.from_wav(path)
    start_ms = int(start_sec * 1000)
    if duration_sec is not None:
        end_ms = start_ms + int(duration_sec * 1000)
        audio = audio[start_ms:end_ms]
    else:
        audio = audio[start_ms:]

    # Export trimmed audio to a temp file, then play with winsound
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    audio.export(tmp.name, format="wav")
    winsound.PlaySound(tmp.name, winsound.SND_FILENAME)
    os.unlink(tmp.name)'''

def play_audio_segment(path, start_sec=0.0, duration_sec=None):
    import wave

    # Try opening; if float32, convert first
    try:
        wave.open(path, 'rb').close()
        wav_path = path
    except wave.Error as e:
        if "unknown format: 3" in str(e):
            wav_path = convert_float32_to_pcm16(path)
        else:
            raise

    with wave.open(wav_path, 'rb') as wf:
        framerate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()

        start_frame = int(start_sec * framerate)
        if duration_sec is not None:
            end_frame = min(start_frame + int(duration_sec * framerate), n_frames)
        else:
            end_frame = n_frames

        wf.setpos(start_frame)
        frames = wf.readframes(end_frame - start_frame)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    with wave.open(tmp.name, 'wb') as out:
        out.setnchannels(n_channels)
        out.setsampwidth(sampwidth)
        out.setframerate(framerate)
        out.writeframes(frames)

    winsound.PlaySound(tmp.name, winsound.SND_FILENAME)
    os.unlink(tmp.name)

def trim_stimulus(audio_path, npz_path, start_sec, end_sec, fps=30):
    """Trim both audio and blendshapes to the same time window."""
    
    # Trim audio
    audio = AudioSegment.from_wav(audio_path)
    trimmed_audio = audio[int(start_sec * 1000):int(end_sec * 1000)]
    
    # Trim blendshapes
    data = np.load(npz_path)
    blendshapes = data['arkit_blendshapes']  # (num_frames, 51)
    
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    trimmed_bs = blendshapes[start_frame:end_frame]
    
    return trimmed_audio, trimmed_bs

furhat = FurhatClient(args.host, auth_key=args.auth_key)
furhat.set_logging_level(logging.INFO)

try:
    furhat.connect()
except Exception as e:
    print(f"Failed to connect to Furhat on {args.host}.")
    exit(0)

furhat.request_attend_location(0, 0, 1)

with open(f"{args.src}") as fid:
    names = next(fid).strip().split()
    threading.Thread(
        target=play_audio_segment,
        args=(args.wav, args.start_sec, args.duration),
        daemon=True
    ).start()
    start = time.time()
    for i, line in enumerate(fid):
        fields = line.split()
        bs = {n: float(e) for n, e in zip(names[:-3], fields[1:-3])}
        furhat.request_face_params(bs)
        bs = {n: float(e) for n, e in zip(names[-3:], fields[-3:])}
        furhat.request_face_headpose(-bs["NECK_PAN"], -bs["NECK_TILT"], bs["NECK_ROLL"], False)
        delay = start + float(fields[0]) - time.time()
        print(i, delay)
        if delay > 0:
            time.sleep(delay)

print("Done")
furhat.disconnect()