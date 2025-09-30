import subprocess
import sys

if len(sys.argv) != 2:
    print("Usage: python split_channels.py input.wav")
    sys.exit(1)

input_file = sys.argv[1]
channel_map = {
    "ch0": "uni_ch0_processed.wav",
    "ch1": "uni_ch1_mic1.wav",
    "ch2": "uni_ch2_mic2.wav",
    "ch3": "uni_ch3_mic3.wav",
    "ch4": "uni_ch4_mic4.wav",
    "ch5": "uni_ch5_playback.wav"
}

cmd = [
    "ffmpeg",
    "-i", input_file,
    "-filter_complex", "[0:a]channelsplit=channel_layout=5.1[ch0][ch1][ch2][ch3][ch4][ch5]"
]

# Append all -map and output file pairs
for ch, fname in channel_map.items():
    cmd += ["-map", f"[{ch}]", fname]

try:
    subprocess.run(cmd, check=True)
    print("Erfolgreich in separate Dateien aufgeteilt.")
except subprocess.CalledProcessError as e:
    print("Fehler beim Ausführen von ffmpeg:", e)

