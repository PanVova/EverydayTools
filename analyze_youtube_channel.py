import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Optional


MAX_CONCURRENT_FETCHES: int = 4
MAX_VIDEOS: int = 15
OUTPUT_DIR: Path = Path.home() / "Downloads" / "youtube_transcripts"
TRANSCRIPT_LANGUAGES: list[str] = ["ru", "en", "en-US"]


@dataclass
class VideoEntry:
    video_id: str
    title: str


@dataclass
class TranscriptEntry:
    video_id: str
    title: str
    text: str


@dataclass
class FetchProgress:
    total: int
    _started: int = field(default=0, repr=False)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def get_next_index(self) -> int:
        with self._lock:
            self._started += 1
            return self._started


@dataclass
class FetchResult:
    successful: int = field(default=0)
    failed: int = field(default=0)
    _lock: Lock = field(default_factory=Lock, repr=False)

    @property
    def total(self) -> int:
        return self.successful + self.failed

    def record_success(self) -> None:
        with self._lock:
            self.successful += 1

    def record_failure(self) -> None:
        with self._lock:
            self.failed += 1

    def print_summary(self) -> None:
        print(f"\n{self.successful}/{self.total} transcripts fetched, {self.failed} failed.")


def install_dependency(package: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def ensure_dependencies() -> None:
    try:
        import yt_dlp
    except ImportError:
        print("Installing yt-dlp...")
        install_dependency("yt-dlp")
    try:
        import youtube_transcript_api
    except ImportError:
        print("Installing youtube-transcript-api...")
        install_dependency("youtube-transcript-api")


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes: int = int(seconds // 60)
    remaining_seconds: int = int(seconds % 60)
    return f"{minutes}m {remaining_seconds}s"


def sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in " _-" else "_" for c in name).strip()


def extract_entries_from_channel(url: str) -> tuple[list[VideoEntry], str]:
    import yt_dlp

    options: dict = {
        "extract_flat": "in_playlist",
        "quiet": True,
        "ignoreerrors": True,
        "playlistend": MAX_VIDEOS,
    }

    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=False)

    if info is None:
        return [], "unknown_channel"

    channel_name: str = info.get("channel") or info.get("uploader") or "unknown_channel"
    entries_raw = info.get("entries", [])
    entries: list[VideoEntry] = [
        VideoEntry(
            video_id=entry.get("id") or "",
            title=entry.get("title") or entry.get("id") or "Unknown",
        )
        for entry in entries_raw
        if entry is not None and entry.get("id")
    ]

    return entries[:MAX_VIDEOS], channel_name


def fetch_transcript_text(video_id: str) -> str:
    from youtube_transcript_api import YouTubeTranscriptApi

    api = YouTubeTranscriptApi()

    try:
        fetched = api.fetch(video_id, languages=TRANSCRIPT_LANGUAGES)
    except Exception:
        fetched = api.fetch(video_id)  # fallback: любой доступный язык

    return " ".join(snippet.text for snippet in fetched)


def fetch_transcript_for_video(
    entry: VideoEntry,
    progress: FetchProgress,
    result: FetchResult,
) -> Optional[TranscriptEntry]:
    index: int = progress.get_next_index()
    print(f"[{index}/{progress.total}] ▶  {entry.title}")

    try:
        text: str = fetch_transcript_text(entry.video_id)
        result.record_success()
        print(f"[{index}/{progress.total}] ✓  {entry.title}")
        return TranscriptEntry(video_id=entry.video_id, title=entry.title, text=text)
    except Exception as error:
        result.record_failure()
        print(f"[{index}/{progress.total}] ✗  {entry.title} — {error}")
        return None


def fetch_all_transcripts_concurrently(entries: list[VideoEntry]) -> list[TranscriptEntry]:
    result = FetchResult()
    progress = FetchProgress(total=len(entries))
    ordered: list[Optional[TranscriptEntry]] = [None] * len(entries)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_FETCHES) as executor:
        future_to_index: dict = {
            executor.submit(fetch_transcript_for_video, entry, progress, result): i
            for i, entry in enumerate(entries)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                ordered[index] = future.result()
            except Exception:
                pass

    result.print_summary()
    return [t for t in ordered if t is not None]


def save_transcripts_to_file(
    transcripts: list[TranscriptEntry],
    channel_name: str,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name: str = sanitize_filename(channel_name)
    output_path: Path = output_dir / f"{safe_name}_transcripts.txt"
    separator: str = "=" * 60

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(f"Channel: {channel_name}\n")
        file.write(f"Videos: {len(transcripts)}\n")
        file.write(f"{separator}\n\n")
        for i, transcript in enumerate(transcripts, start=1):
            file.write(f"{separator}\n")
            file.write(f"[{i}] {transcript.title}\n")
            file.write(f"URL: https://www.youtube.com/watch?v={transcript.video_id}\n")
            file.write(f"{separator}\n\n")
            file.write(transcript.text)
            file.write("\n\n")

    return output_path


def collect_url() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    return input("Enter YouTube channel URL: ").strip()


def main() -> None:
    ensure_dependencies()

    url: str = collect_url()
    if not url:
        print("Error: no URL provided.")
        sys.exit(1)

    start_time: float = time.perf_counter()

    print("Extracting video list from channel...")
    entries, channel_name = extract_entries_from_channel(url)

    if not entries:
        print("Error: no videos found.")
        sys.exit(1)

    print(f"Channel: {channel_name}")
    print(f"Found {len(entries)} video(s). Fetching transcripts...\n")

    transcripts: list[TranscriptEntry] = fetch_all_transcripts_concurrently(entries)

    if not transcripts:
        print("Error: no transcripts could be fetched.")
        sys.exit(1)

    output_path: Path = save_transcripts_to_file(transcripts, channel_name, OUTPUT_DIR)
    print(f"\nSaved to: {output_path}")

    elapsed: float = time.perf_counter() - start_time
    print(f"Total time: {format_duration(elapsed)}")


if __name__ == "__main__":
    main()