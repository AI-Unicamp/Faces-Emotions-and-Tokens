import numpy as np
import wave
import struct

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


def add_noise_to_wav(input_path, output_path, snr_db):
    """
    Add speech-shaped noise to a WAV file at a target SNR.
    
    Args:
        input_path:  clean WAV file
        output_path: noisy WAV output
        snr_db:      signal-to-noise ratio in dB
                     0 dB  = equal loudness (moderate)
                     -6 dB = noise louder (high noise)
    """

    # Convert float32 to pcm16 if needed
    try:
        wave.open(input_path, 'rb').close()
        wav_path = input_path
    except wave.Error as e:
        if "unknown format: 3" in str(e):
            wav_path = convert_float32_to_pcm16(input_path)
        else:
            raise
    # Read audio

    with wave.open(wav_path, 'rb') as wf:
        params = wf.getparams()
        frames = wf.readframes(params.nframes)
    
    if params.sampwidth == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float64)
    elif params.sampwidth == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float64)
    else:
        raise ValueError(f"Unsupported sample width: {params.sampwidth}")
    
    # Generate white noise
    noise = np.random.randn(len(audio))
    
    # Shape it to match speech spectrum (simple low-pass approximation)
    # Speech-shaped noise: roll off above ~1kHz
    from scipy.signal import butter, lfilter
    nyquist = params.framerate / 2
    # Speech energy is concentrated below 4kHz, with rolloff above 1kHz
    b, a = butter(4, [100 / nyquist, 4000 / nyquist], btype='band')
    noise = lfilter(b, a, noise)
    
    # Scale noise to target SNR
    signal_power = np.mean(audio ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        raise ValueError("Noise power is zero")
    
    # SNR = 10 * log10(signal_power / noise_power)
    # noise_scale = sqrt(signal_power / (noise_power * 10^(snr_db/10)))
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noise_scale = np.sqrt(target_noise_power / noise_power)
    noise = noise * noise_scale
    
    # Mix
    mixed = audio + noise
    
    # Normalize to same RMS as original clean audio
    original_rms = np.sqrt(np.mean(audio ** 2))
    mixed_rms = np.sqrt(np.mean(mixed ** 2))
    if mixed_rms > 0:
        mixed = mixed * (original_rms / mixed_rms)
    
    # Clip to prevent overflow
    if params.sampwidth == 2:
        mixed = np.clip(mixed, -32768, 32767).astype(np.int16)
    else:
        mixed = np.clip(mixed, -2147483648, 2147483647).astype(np.int32)
    
    # Write
    with wave.open(output_path, 'wb') as wf:
        wf.setparams(params)
        wf.writeframes(mixed.tobytes())


def generate_noise_conditions(clean_wav_path, output_dir):
    """Generate all noise conditions for a stimulus."""
    import shutil
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stem = Path(clean_wav_path).stem
    
    # No noise — just copy
    no_noise_path = output_dir / f"{stem}_clean.wav"
    shutil.copy2(clean_wav_path, no_noise_path)
    
    # Moderate noise: 0 dB SNR
    moderate_path = output_dir / f"{stem}_noise_0dB.wav"
    add_noise_to_wav(clean_wav_path, str(moderate_path), snr_db=0)
    
    # High noise: -6 dB SNR
    high_path = output_dir / f"{stem}_noise_-6dB.wav"
    add_noise_to_wav(clean_wav_path, str(high_path), snr_db=-6)
    
    print(f"Generated:")
    print(f"  Clean:    {no_noise_path}")
    print(f"  0 dB SNR: {moderate_path}")
    print(f"  -6 dB SNR: {high_path}")
    
    return no_noise_path, moderate_path, high_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Clean WAV file")
    parser.add_argument("--output_dir", default="noisy_stimuli/")
    parser.add_argument("--snr", type=float, nargs='+', default=[0, -6],
                        help="SNR levels in dB (default: 0 -6)")
    args = parser.parse_args()
    
    if args.snr == [0, -6]:
        generate_noise_conditions(args.input, args.output_dir)
    else:
        from pathlib import Path
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        stem = Path(args.input).stem
        for snr in args.snr:
            out = f"{args.output_dir}/{stem}_noise_{snr}dB.wav"
            add_noise_to_wav(args.input, out, snr_db=snr)
            print(f"  {snr} dB SNR: {out}")



'''

python add_noise.py --input data/perceptual_test_4/george_ada.wav --output_dir data/perceptual_test_4/george_ada_noise_3 --snr -6 -13

python add_noise.py --input stimuli/ahba_clean.wav --output_dir stimuli/noisy/

'''
