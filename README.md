
Dieses Repository enthält Skripte zur Aufnahme, Kanal-Trennung und Sensorabfrage mit dem ReSpeaker 4-Mic Array v2.0 (USB) sowie zur Auswertung von VAD (Voice Activity Detection) und DoA (Direction of Arrival).
Voraussetzungen (je nach Skript): python3, pyaudio, ffmpeg, pyusb, sowie das Seeed-Studio tuning.py-Modul.

Skripte

record.py
Nimmt Mehrkanal-Audio (z. B. 6 Kanäle bei der Factory-Firmware) vom ReSpeaker über PyAudio auf und speichert es als WAV (Samplerate/Channels/Dauer im Skript konfigurierbar).

split_channel.py
Teilt eine zuvor aufgenommene 6-Kanal-WAV mittels ffmpeg in einzelne Monodateien auf und benennt sie entsprechend der ReSpeaker-Belegung
(ch0 = processed/ASR, ch1..ch4 = Mic1..Mic4 raw, ch5 = Playback/Mix).

VAD.py
Fragt über USB (PyUSB + tuning.py) das VAD-Flag des Arrays ab und gibt einmal pro Sekunde aus, ob gerade Stimme erkannt wird.

DOA.py
Liest über USB (PyUSB + tuning.py) kontinuierlich die geschätzte Einfallsrichtung (DoA) der dominanten Schallquelle aus und schreibt den Winkel in die Konsole.

get_index.py
Hinweis: Die hier mitgelieferte Datei enthält aktuell denselben Code wie record.py (Aufnahme).
Zwecklich sollte get_index.py die Audio-Geräteliste über PyAudio ausgeben, um die Device-ID (Index) des ReSpeaker zu finden.
Empfehlung: Ersetze den Inhalt durch ein kurzes Listing-Skript, das alle Input-Devices mit Index und Namen ausgibt, damit du RESPEAKER_INDEX in record.py korrekt setzen kannst.

Abhängigkeiten

System: ALSA (Linux), ffmpeg

Python: pyaudio, pyusb, optional numpy (falls benötigt), sowie das Seeed-tuning.py (im selben Ordner wie VAD.py/DOA.py)

ReSpeaker 4-Mic Array v2.0 (USB, Produkt-ID 2886:0018)

Beispiel-Nutzung

Geräteindex ermitteln
(siehe Hinweis zu get_index.py – danach RESPEAKER_INDEX in record.py setzen)

Aufnahme starten
