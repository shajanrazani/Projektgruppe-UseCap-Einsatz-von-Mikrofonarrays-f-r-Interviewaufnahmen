#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gesamt-Skript: Transkription (Whisper) + Sprecherdiarisierung (pyannote)
mit robustem Merging zusammenhängender Segmente und Pfad-Preflight.

Voraussetzungen (pip):
    pip install openai-whisper pyannote.audio torch torchaudio
"""
#Alle Parameter sollten an aktuelles Projekt angepasst werden

import os
import argparse
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

import torch
import whisper
from pyannote.audio import Pipeline


# === DEFAULTS ===
AUDIO_FILE = r"Pfad zur Audiodatei"
OUTPUT_FILE = "Output name"
WHISPER_MODEL_SIZE = "large"
GAP_THRESH = 0.75 #Anpassen!
HUGGINGFACE_TOKEN = "hier Hugginface Token einfügen"



# === CLI ===
def parse_args():
    p = argparse.ArgumentParser(description="Transkription + Diarisierung mit Merging.")
    p.add_argument("--audio", type=str, default=AUDIO_FILE, help="Pfad zur Audiodatei")
    p.add_argument("--out", type=str, default=OUTPUT_FILE, help="Zieldatei für Transkript")
    p.add_argument("--model", type=str, default=WHISPER_MODEL_SIZE, help="Whisper-Modellgröße")
    p.add_argument("--gap", type=float, default=GAP_THRESH, help="Merge-Lücke in Sekunden")
    p.add_argument("--hf_token", type=str, default=HUGGINGFACE_TOKEN, help="Hugging Face Token")
    return p.parse_args()


@dataclass
class Utterance:
    start: float
    end: float
    speaker: str
    text: str


# === AUDIO PREP ===
def prepare_audio_path(audio_path: str) -> str:
    p = Path(audio_path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Audio nicht gefunden: {audio_path}")
    tmpdir = Path(tempfile.gettempdir()) / "whisper_in"
    tmpdir.mkdir(exist_ok=True)
    short = tmpdir / "audio.wav"
    shutil.copy2(p, short)
    print(f"✅ Audio bereit: {short}")
    return str(short)


# === WHISPER ===
def load_whisper(model_size: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⏱️  Lade Whisper-Modell ({model_size}) auf {device.upper()}...")
    return whisper.load_model(model_size, device=device)


def transcribe(model, audio_path: str):
    print(f"🎙️  Transkribiere: {audio_path}")
    result = model.transcribe(audio_path, language=None, verbose=False, word_timestamps=False)
    segments = result.get("segments", [])
    norm_segments = []
    for s in segments:
        start = float(s.get("start", 0.0))
        end = float(s.get("end", 0.0))
        text = (s.get("text") or "").strip()
        if text:
            norm_segments.append({"start": start, "end": end, "text": text})
    print(f"   → {len(norm_segments)} Whisper-Segmente")
    return norm_segments


# === DIARIZATION ===
def run_diarization(audio_path: str, hf_token: str):
    if not hf_token:
        raise RuntimeError("Bitte Hugging Face Token über --hf_token oder HUGGINGFACE_TOKEN setzen.")
    print("🗣️  Lade pyannote-Pipeline (speaker-diarization)...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)
    diarization = pipeline(audio_path)
    print("   → Diarisierung abgeschlossen")
    dia_segments: List[Tuple[float, float, str]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        dia_segments.append((float(turn.start), float(turn.end), speaker))
    return dia_segments


# === HELPER ===
def best_speaker_for_interval(dia_segments: List[Tuple[float, float, str]], t0: float, t1: float) -> str:
    best_label, best_ovlp = "Unknown", 0.0
    for s0, s1, spk in dia_segments:
        ovlp = max(0.0, min(t1, s1) - max(t0, s0))
        if ovlp > best_ovlp:
            best_ovlp = ovlp
            best_label = spk
    return best_label


def merge_utterances(segments: List[dict], dia_segments: List[Tuple[float, float, str]], gap_thresh: float) -> List[Utterance]:
    utterances: List[Utterance] = []
    for seg in segments:
        start, end, text = float(seg["start"]), float(seg["end"]), seg["text"].strip()
        spk = best_speaker_for_interval(dia_segments, start, end)
        utterances.append(Utterance(start, end, spk, text))

    merged: List[Utterance] = []
    for u in utterances:
        if merged and u.speaker == merged[-1].speaker and u.start <= merged[-1].end + gap_thresh:
            merged[-1].end = max(merged[-1].end, u.end)
            sep = "" if merged[-1].text.endswith(("-", "–", "—", " ")) else " "
            if u.text and (u.text not in merged[-1].text[-len(u.text)-2:]):
                merged[-1].text = (merged[-1].text + sep + u.text).strip()
        else:
            merged.append(Utterance(u.start, u.end, u.speaker, u.text))
    return merged


def save_transcript(utterances: List[Utterance], out_path: str):
    lines = [f"[{u.start:.2f} – {u.end:.2f}] {u.speaker}: {u.text}" for u in utterances]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"✅ Transkript gespeichert in '{out_path}'")


# === MAIN ===
def main():
    args = parse_args()
    safe_audio = prepare_audio_path(args.audio)
    model = load_whisper(args.model)
    segments = transcribe(model, safe_audio)
    dia_segments = run_diarization(safe_audio, args.hf_token)
    merged = merge_utterances(segments, dia_segments, args.gap)
    save_transcript(merged, args.out)


if __name__ == "__main__":
    main()
