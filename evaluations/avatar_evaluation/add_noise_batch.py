import os
import glob
from add_noise import add_noise_to_wav

# --- CONFIGURATION ---
input_folder = "results/I_I_samples"
output_folder = "results/I_I_samples_noise"
snr_levels = [-6, -13, -30]  # Add the dB levels you want here
# ---------------------

os.makedirs(output_folder, exist_ok=True)

# Find all wav files
wav_files = glob.glob(os.path.join(input_folder, "*.wav"))

for wav_path in wav_files:
    filename = os.path.basename(wav_path)
    stem = os.path.splitext(filename)[0]
    
    print(f"Processing: {filename}")
    
    for snr in snr_levels:
        output_name = f"{stem}_noise_{snr}dB.wav"
        output_path = os.path.join(output_folder, output_name)
        
        try:
            add_noise_to_wav(wav_path, output_path, snr_db=snr)
            print(f"  -> Generated {snr}dB version")
        except Exception as e:
            print(f"  -> Error processing {filename} at {snr}dB: {e}")

print("\nDone! All files processed.")