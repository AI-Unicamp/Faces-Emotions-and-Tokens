#!/usr/bin/env python3
"""
Furhat Facial Animation Evaluation Platform
Forced-choice intelligibility test: user watches Furhat, picks what syllable was said.

File naming convention for samples:
    samples/<SYLLABLE>_<NOISE>.txt
    samples/<SYLLABLE>_<NOISE>.wav

    SYLLABLE:  ABA, ACHA, ADA, AFA
    NOISE:     noise1, noise2, noise3

Example:
    samples/ABA_noise1.txt
    samples/ABA_noise1.wav
"""

from flask import Flask, render_template, jsonify, request, session, send_from_directory
import os
import sys
import json
import csv
import random
import time
import uuid
import struct
import wave
import tempfile
import threading
from glob import glob

from furhat_realtime_api import FurhatClient
import numpy as np
import logging

# ---------------------------------------------------------------------------
# Configuration (overridden by CLI args)
# ---------------------------------------------------------------------------
FURHAT_HOST = os.environ.get("FURHAT_HOST", "127.0.0.1")
WAV_WEB_PREFIX = ""  # set at startup based on Flask's own address
SIM_MODE = False

furhat = None
furhat_lock = threading.Lock()
playback_in_progress = False


# ---------------------------------------------------------------------------
# Furhat connection
# ---------------------------------------------------------------------------
def connect_furhat():
    global furhat
    furhat = FurhatClient(FURHAT_HOST, auth_key=None)
    furhat.set_logging_level(logging.INFO)
    try:
        furhat.connect()
    except Exception as e:
        print(f"[ERROR] Failed to connect to Furhat on {FURHAT_HOST}: {e}")
        furhat = None
        return False
    furhat.request_attend_location(0, 0, 1)
    print(f"[OK] Connected to Furhat at {FURHAT_HOST}")
    return True


def disconnect_furhat():
    global furhat
    if furhat:
        furhat.disconnect()
        furhat = None
        print("[OK] Disconnected from Furhat")


# ---------------------------------------------------------------------------
# Local audio playback (for simulated Furhat on Windows)
# ---------------------------------------------------------------------------
def convert_float32_to_pcm16(path):
    """Convert float32 WAV to pcm_s16le WAV."""
    with open(path, 'rb') as f:
        riff = f.read(4)
        if riff != b'RIFF':
            raise ValueError("Not a WAV file")
        f.read(4)  # file size
        f.read(4)  # WAVE
        fmt_data = audio_data = None
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
    samples = np.frombuffer(audio_data, dtype=np.float32)
    samples = np.clip(samples, -1.0, 1.0)
    samples_int16 = (samples * 32767).astype(np.int16)

    out_path = path.replace(".wav", "_pcm16.wav")
    with wave.open(out_path, 'wb') as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples_int16.tobytes())
    print(f"[OK] Converted float32→pcm16: {out_path}")
    return out_path


def play_audio_local(path, start_sec=0.0, duration_sec=None):
    """Play a wav file locally using winsound (Windows) or aplay (Linux)."""
    # Ensure PCM format
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
        n_frames = wf.getnframes()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        start_frame = int(start_sec * framerate)
        end_frame = min(start_frame + int(duration_sec * framerate), n_frames) \
            if duration_sec else n_frames
        wf.setpos(start_frame)
        frames = wf.readframes(end_frame - start_frame)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    with wave.open(tmp.name, 'wb') as out:
        out.setnchannels(n_channels)
        out.setsampwidth(sampwidth)
        out.setframerate(framerate)
        out.writeframes(frames)

    if sys.platform == "win32":
        import winsound
        winsound.PlaySound(tmp.name, winsound.SND_FILENAME)
    else:
        os.system(f"aplay {tmp.name} 2>/dev/null")

    os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# Playback: real Furhat (HTTP audio) vs simulated (local audio)
# ---------------------------------------------------------------------------
# Blendshape names (excluding neck) used to reset face to neutral
_FACE_AU_NAMES = (
    "BROW_INNER_UP BROW_OUTER_UP_LEFT BROW_OUTER_UP_RIGHT CHEEK_PUFF CHEEK_SQUINT_LEFT CHEEK_SQUINT_RIGHT "
    "EYE_BLINK_LEFT EYE_BLINK_RIGHT EYE_LOOK_DOWN_LEFT EYE_LOOK_DOWN_RIGHT EYE_LOOK_IN_LEFT EYE_LOOK_IN_RIGHT EYE_LOOK_OUT_LEFT "
    "EYE_LOOK_OUT_RIGHT EYE_LOOK_UP_LEFT EYE_LOOK_UP_RIGHT EYE_SQUINT_LEFT EYE_SQUINT_RIGHT "
    "EYE_WIDE_LEFT EYE_WIDE_RIGHT JAW_FORWARD JAW_LEFT JAW_OPEN JAW_RIGHT MOUTH_CLOSE MOUTH_DIMPLE_LEFT MOUTH_DIMPLE_RIGHT "
    "MOUTH_FROWN_LEFT MOUTH_FROWN_RIGHT MOUTH_FUNNEL MOUTH_LEFT MOUTH_LOWER_DOWN_LEFT MOUTH_LOWER_DOWN_RIGHT "
    "MOUTH_PRESS_LEFT MOUTH_PRESS_RIGHT MOUTH_PUCKER MOUTH_RIGHT MOUTH_ROLL_LOWER MOUTH_ROLL_UPPER "
    "MOUTH_SHRUG_LOWER MOUTH_SHRUG_UPPER MOUTH_SMILE_LEFT MOUTH_SMILE_RIGHT MOUTH_STRETCH_LEFT MOUTH_STRETCH_RIGHT "
    "MOUTH_UPPER_UP_LEFT MOUTH_UPPER_UP_RIGHT NOSE_SNEER_LEFT NOSE_SNEER_RIGHT"
).split()


def reset_face():
    """Reset all blendshapes to zero and head pose to neutral."""
    if furhat is None:
        return
    furhat.request_face_params({name: 0.0 for name in _FACE_AU_NAMES})
    furhat.request_face_headpose(0, 0, 0, False)


def play_on_furhat_real(sample):
    """Real Furhat: send audio via HTTP + stream blendshapes from .txt."""
    global playback_in_progress
    with furhat_lock:
        playback_in_progress = True
        try:
            wav_url = f"{sample['wav_url']}?t={int(time.time())}"
            with open(sample["txt_path"]) as fid:
                names = next(fid).strip().split()
                furhat.request_speak_audio(
                    url=wav_url, wait=False, abort=True,
                    text="eval_audio", lipsync=False,
                )
                start = time.time()
                for line in fid:
                    fields = line.split()
                    ts = float(fields[0])
                    values = fields[1:]
                    bs = {n: float(v) for n, v in zip(names[:-3], values[:-3])}
                    furhat.request_face_params(bs)
                    neck = {n: float(v) for n, v in zip(names[-3:], values[-3:])}
                    furhat.request_face_headpose(
                        -neck["NECK_PAN"], -neck["NECK_TILT"], neck["NECK_ROLL"], False
                    )
                    delay = start + ts - time.time()
                    if delay > 0:
                        time.sleep(delay)
        finally:
            playback_in_progress = False


def play_on_furhat_sim(sample):
    """Simulated Furhat: stream blendshapes + play audio locally."""
    global playback_in_progress
    with furhat_lock:
        playback_in_progress = True
        try:
            with open(sample["txt_path"]) as fid:
                names = next(fid).strip().split()
                audio_thread = threading.Thread(
                    target=play_audio_local,
                    args=(sample["wav_path"],),
                    daemon=True,
                )
                audio_thread.start()
                start = time.time()
                for line in fid:
                    fields = line.split()
                    ts = float(fields[0])
                    values = fields[1:]
                    bs = {n: float(v) for n, v in zip(names[:-3], values[:-3])}
                    furhat.request_face_params(bs)
                    neck = {n: float(v) for n, v in zip(names[-3:], values[-3:])}
                    furhat.request_face_headpose(
                        -neck["NECK_PAN"], -neck["NECK_TILT"], neck["NECK_ROLL"], False
                    )
                    delay = start + ts - time.time()
                    if delay > 0:
                        time.sleep(delay)
                audio_thread.join(timeout=5)
        finally:
            playback_in_progress = False


def play_sample(sample):
    """Dispatch to the right playback function based on mode."""
    if SIM_MODE:
        play_on_furhat_sim(sample)
    else:
        play_on_furhat_real(sample)
    reset_face()


# ---------------------------------------------------------------------------
# Sample discovery
# ---------------------------------------------------------------------------
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")
SYLLABLES = ["ABA", "ACHA", "ADA", "AFA"]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def discover_samples():
    """Find all valid sample pairs (.txt + .wav) in samples/ folder.
    Naming: <SYLLABLE>_<NOISE>.txt + .wav  (e.g. ABA_noise1.txt)
    """
    samples = []
    wav_files = glob(os.path.join(SAMPLES_DIR, "*.wav"))
    for wav_path in sorted(wav_files):
        base = os.path.splitext(os.path.basename(wav_path))[0]

        parts = base.split("_")
        if len(parts) < 2:
            print(f"[WARN] Unexpected filename format: {base}, skipping")
            continue

        syllable = parts[0]
        noise = "_".join(parts[1:])

        if syllable not in SYLLABLES:
            print(f"[WARN] Unknown syllable '{syllable}' in {base}, skipping")
            continue

        txt_path = os.path.join(SAMPLES_DIR, base + ".txt")
        if not os.path.isfile(txt_path):
            print(f"[WARN] Missing txt for {base}, skipping")
            continue

        samples.append({
            "id": base,
            "syllable": syllable,
            "noise": noise,
            "txt_path": txt_path,
            "wav_path": wav_path,
            "wav_url": WAV_WEB_PREFIX + base + ".wav",
        })

    print(f"[OK] Discovered {len(samples)} samples")
    return samples


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------
def save_result(user_id, trial):
    """Append one trial result to both CSV and JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    csv_path = os.path.join(RESULTS_DIR, "results.csv")
    write_header = not os.path.isfile(csv_path)
    row = {
        "user_id": user_id,
        "timestamp": trial["timestamp"],
        "trial_index": trial["trial_index"],
        "sample_id": trial["sample_id"],
        "syllable": trial["syllable"],
        "noise": trial["noise"],
        "response": trial["response"],
        "correct": trial["correct"],
        "reaction_time_ms": trial["reaction_time_ms"],
    }
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    json_path = os.path.join(RESULTS_DIR, f"user_{user_id}.json")
    if os.path.isfile(json_path):
        with open(json_path) as f:
            data = json.load(f)
    else:
        data = {"user_id": user_id, "trials": []}
    data["trials"].append(trial)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "furhat-eval-secret-key")

ALL_SAMPLES = []  # populated on startup


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/samples/")
def list_samples():
    """List all files in samples/ for debugging."""
    files = sorted(os.listdir(SAMPLES_DIR))
    listing = "\n".join(f'<a href="/samples/{f}">{f}</a>' for f in files)
    return f"<pre>{listing}</pre>"


@app.route("/samples/<path:filename>")
def serve_sample(filename):
    """Serve wav files from samples/ so the Furhat can fetch them via HTTP."""
    return send_from_directory(SAMPLES_DIR, filename)


@app.route("/api/start", methods=["POST"])
def api_start():
    """Initialize a new evaluation session with randomized sample order."""
    user_id = str(uuid.uuid4())[:8]
    order = list(range(len(ALL_SAMPLES)))
    random.shuffle(order)
    session["user_id"] = user_id
    session["order"] = order
    session["current"] = 0
    session["started_at"] = time.time()
    return jsonify({
        "user_id": user_id,
        "total_trials": len(order),
    })


@app.route("/api/play", methods=["POST"])
def api_play():
    """Play the current sample on the Furhat robot."""
    if playback_in_progress:
        return jsonify({"error": "Playback already in progress"}), 409

    idx = session.get("current", 0)
    order = session.get("order", [])
    if idx >= len(order):
        return jsonify({"error": "No more samples"}), 400

    sample = ALL_SAMPLES[order[idx]]

    def _play():
        try:
            play_sample(sample)
        except Exception as e:
            print(f"[ERROR] Playback failed: {e}")

    t = threading.Thread(target=_play, daemon=True)
    t.start()

    return jsonify({
        "status": "playing",
        "trial_index": idx + 1,
        "total_trials": len(order),
        "play_started_at": time.time(),
    })


@app.route("/api/respond", methods=["POST"])
def api_respond():
    """Record the user's response for the current trial."""
    if playback_in_progress:
        return jsonify({"error": "Wait for playback to finish"}), 409

    data = request.json
    response = data.get("response")
    reaction_time_ms = data.get("reaction_time_ms", 0)

    if response not in SYLLABLES:
        return jsonify({"error": f"Invalid response: {response}"}), 400

    idx = session.get("current", 0)
    order = session.get("order", [])
    user_id = session.get("user_id", "unknown")

    if idx >= len(order):
        return jsonify({"error": "No more samples"}), 400

    sample = ALL_SAMPLES[order[idx]]
    correct = response == sample["syllable"]

    trial = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trial_index": idx + 1,
        "sample_id": sample["id"],
        "syllable": sample["syllable"],
        "noise": sample["noise"],
        "response": response,
        "correct": correct,
        "reaction_time_ms": reaction_time_ms,
    }

    save_result(user_id, trial)

    session["current"] = idx + 1
    done = (idx + 1) >= len(order)

    return jsonify({
        "trial_index": idx + 1,
        "total_trials": len(order),
        "done": done,
    })


@app.route("/api/status", methods=["GET"])
def api_status():
    """Check current session status."""
    return jsonify({
        "user_id": session.get("user_id"),
        "current": session.get("current", 0),
        "total": len(session.get("order", [])),
        "playback_in_progress": playback_in_progress,
        "furhat_connected": furhat is not None,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import socket

    def get_local_ip():
        """Get this machine's LAN IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    parser = argparse.ArgumentParser(description="Furhat Evaluation Platform")
    parser.add_argument("--host", default="127.0.0.1", help="Furhat robot IP/hostname")
    parser.add_argument("--port", type=int, default=5000, help="Flask server port")
    parser.add_argument("--sim", action="store_true",
                        help="Simulated Furhat mode: audio plays locally via winsound/aplay")
    parser.add_argument("--no-furhat", action="store_true",
                        help="Run without any Furhat connection (UI testing only)")
    args = parser.parse_args()

    FURHAT_HOST = args.host
    SIM_MODE = args.sim

    # Flask serves wav files itself — build URL prefix using this machine's LAN IP
    local_ip = get_local_ip()
    WAV_WEB_PREFIX = f"http://{local_ip}:{args.port}/samples/"
    print(f"[OK] Serving wav files at {WAV_WEB_PREFIX}")

    mode_label = "SIMULATED" if SIM_MODE else "REAL"

    ALL_SAMPLES = discover_samples()
    if not ALL_SAMPLES:
        print("[ERROR] No samples found in samples/ directory.")
        print(f"        Expected files like: {SAMPLES_DIR}/ABA_noise1.txt + .wav")
        exit(1)

    if not args.no_furhat:
        if not connect_furhat():
            print(f"[ERROR] Could not connect to Furhat at {FURHAT_HOST}.")
            print("        Use --no-furhat to test UI only.")
            exit(1)
        print(f"[OK] Mode: {mode_label} (audio {'played locally' if SIM_MODE else 'fetched by robot via HTTP'})")
    else:
        print("[OK] Mode: NO FURHAT (UI testing only)")

    try:
        print(f"\n  Open http://localhost:{args.port} in your browser\n")
        app.run(host="0.0.0.0", port=args.port, debug=False)
    finally:
        disconnect_furhat()