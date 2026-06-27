

import os
import sys
import tempfile
from pathlib import Path

import torch
import yt_dlp
import whisper


# ── Constants ──────────────────────────────────────────────────────────────────

SUPPORTED_MEDIA_EXTENSIONS: set[str] = {
    ".mp4", ".mkv", ".avi", ".mov", ".webm",
    ".mp3", ".wav", ".m4a", ".flac", ".ogg",
}
DEFAULT_MODEL: str = "medium"
AUDIO_FORMAT: str = "mp3"
OUTPUT_TRANSCRIPT_EXTENSION: str = ".txt"
SCRIPT_DIRECTORY: Path = Path(__file__).parent.resolve()


# ── GPU verification ───────────────────────────────────────────────────────────

def resolveDevice() -> str:
    """
    Detect available compute device and print a summary.

    Returns:
        'cuda' if a CUDA-capable GPU is available, otherwise 'cpu'.
    """
    if not torch.cuda.is_available():
        print("[device] ⚠  No CUDA GPU detected — running on CPU (will be slow).")
        return "cpu"

    deviceName: str = torch.cuda.get_device_name(0)
    totalVram: float = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"[device] ✓  GPU detected: {deviceName} ({totalVram:.1f} GB VRAM)")

    minimumVramForLargeModel: float = 9.0
    if totalVram < minimumVramForLargeModel:
        print(
            f"[device] ⚠  {totalVram:.1f} GB VRAM may be insufficient for '{DEFAULT_MODEL}' "
            f"(recommended ≥ {minimumVramForLargeModel:.0f} GB). "
            "Consider switching to 'medium'."
        )

    return "cuda"


# ── File discovery ─────────────────────────────────────────────────────────────

def findMediaFiles(directory: Path) -> list[Path]:
    """
    Scan a directory for supported media files (non-recursive).

    Args:
        directory: Directory to scan.

    Returns:
        Sorted list of matching file paths.
    """
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_MEDIA_EXTENSIONS
    )


def selectMediaFile(directory: Path) -> Path:
    """
    Auto-select a media file from the script directory.

    - If exactly one file is found, it is selected automatically.
    - If multiple files are found, the user picks one by number.
    - If none are found, exits with an error.

    Args:
        directory: Directory to scan for media files.

    Returns:
        Selected media file path.
    """
    mediaFiles: list[Path] = findMediaFiles(directory)

    if not mediaFiles:
        print(
            f"[error] No supported media files found in: {directory}\n"
            f"        Supported formats: {', '.join(sorted(SUPPORTED_MEDIA_EXTENSIONS))}",
            file=sys.stderr,
        )
        sys.exit(1)

    if len(mediaFiles) == 1:
        print(f"[file] Auto-selected: {mediaFiles[0].name}")
        return mediaFiles[0]

    print("[file] Multiple media files found — pick one:")
    for index, path in enumerate(mediaFiles, start=1):
        print(f"  {index}. {path.name}")

    choice: str = input("Enter number: ").strip()
    if not choice.isdigit() or not (1 <= int(choice) <= len(mediaFiles)):
        print("[error] Invalid selection.", file=sys.stderr)
        sys.exit(1)

    return mediaFiles[int(choice) - 1]


# ── Transcriber ────────────────────────────────────────────────────────────────

def loadWhisperModel(modelName: str, device: str) -> whisper.Whisper:
    """
    Load a Whisper model onto the specified device.

    Args:
        modelName: Whisper model size (e.g. 'large-v3').
        device: 'cuda' or 'cpu'.

    Returns:
        Loaded Whisper model.
    """
    print(f"[whisper] Loading model '{modelName}' on {device} ...")
    return whisper.load_model(modelName, device=device)


def transcribeAudio(
    model: whisper.Whisper,
    audioPath: str,
    language: str | None = None,
) -> dict:
    """
    Run Whisper transcription on an audio file.

    Args:
        model: Loaded Whisper model.
        audioPath: Path to the audio file.
        language: ISO-639-1 code to force language; None = auto-detect.

    Returns:
        Whisper result dict with keys: text, segments, language.
    """
    print(f"[whisper] Transcribing: {audioPath}")
    transcribeArgs: dict = {"verbose": True}
    if language:
        transcribeArgs["language"] = language
    return model.transcribe(audioPath, **transcribeArgs)


# ── Output ─────────────────────────────────────────────────────────────────────

def saveTranscript(text: str, outputPath: Path) -> None:
    """Write transcript text to a .txt file next to the source."""
    outputPath.write_text(text, encoding="utf-8")
    print(f"[output] Transcript saved → {outputPath}")


def printTimestampedTranscript(segments: list[dict]) -> None:
    """Print transcript lines with timestamps to stdout."""
    print("\n── Timestamped Transcript ──────────────────────────────────────")
    for segment in segments:
        start: str = formatTimestamp(segment["start"])
        end: str = formatTimestamp(segment["end"])
        text: str = segment["text"].strip()
        print(f"[{start} → {end}]  {text}")
    print("────────────────────────────────────────────────────────────────\n")


def formatTimestamp(seconds: float) -> str:
    """Convert float seconds to HH:MM:SS string."""
    totalSeconds: int = int(seconds)
    hours: int = totalSeconds // 3600
    minutes: int = (totalSeconds % 3600) // 60
    secs: int = totalSeconds % 60
    return f"{hours:02}:{minutes:02}:{secs:02}"


# ── Orchestration ──────────────────────────────────────────────────────────────

def processFile(
    sourceFile: Path,
    model: whisper.Whisper,
    language: str | None = None,
    showTimestamps: bool = True,
) -> str:
    """
    Transcribe a local media file and save the result alongside it.

    Args:
        sourceFile: Path to the media file.
        model: Loaded Whisper model.
        language: Force language; None = auto-detect.
        showTimestamps: Print timestamped segments to stdout.

    Returns:
        Full transcript text.
    """
    result: dict = transcribeAudio(model, str(sourceFile), language)
    transcriptText: str = result["text"].strip()
    detectedLanguage: str = result.get("language", "unknown")

    print(f"[info] Detected language: {detectedLanguage}")

    if showTimestamps:
        printTimestampedTranscript(result.get("segments", []))

    outputPath: Path = sourceFile.with_suffix(OUTPUT_TRANSCRIPT_EXTENSION)
    saveTranscript(transcriptText, outputPath)

    return transcriptText


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    device: str = resolveDevice()
    sourceFile: Path = selectMediaFile(SCRIPT_DIRECTORY)
    model: whisper.Whisper = loadWhisperModel(DEFAULT_MODEL, device)

    try:
        transcript: str = processFile(
            sourceFile=sourceFile,
            model=model,
            language=None,      # None = auto-detect; set e.g. "en" to force
            showTimestamps=True,
        )
        print(f"\n[done] Transcript length: {len(transcript)} characters.")
    except KeyboardInterrupt:
        print("\n[cancelled] Interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
