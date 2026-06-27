import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock


MAX_CONCURRENT_DOWNLOADS: int = 4
OUTPUT_DIR: Path = Path.home() / "Downloads" / "youtube_mp3"
AUDIO_FORMAT: str = "mp3"
AUDIO_QUALITY: str = "192"


@dataclass
class VideoEntry:
    url: str
    title: str


@dataclass
class DownloadProgress:
    total: int
    _started: int = field(default=0, repr=False)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def get_next_index(self) -> int:
        with self._lock:
            self._started += 1
            return self._started


@dataclass
class DownloadResult:
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
        print(f"\n{self.successful}/{self.total} downloaded, {self.failed} failed.")


class SilentLogger:
    def debug(self, message: str) -> None:
        pass

    def warning(self, message: str) -> None:
        pass

    def error(self, message: str) -> None:
        pass


def install_dependency(package: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def ensure_dependencies() -> None:
    try:
        import yt_dlp
    except ImportError:
        print("Installing yt-dlp...")
        install_dependency("yt-dlp")


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes: int = int(seconds // 60)
    remaining_seconds: int = int(seconds % 60)
    return f"{minutes}m {remaining_seconds}s"


def build_single_download_options(output_dir: Path) -> dict:
    return {
        "format": "bestaudio/best",
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
        "logger": SilentLogger(),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": AUDIO_FORMAT,
                "preferredquality": AUDIO_QUALITY,
            }
        ],
    }


def extract_entries_from_url(url: str) -> list[VideoEntry]:
    import yt_dlp

    options: dict = {
        "extract_flat": "in_playlist",
        "quiet": True,
        "ignoreerrors": True,
    }

    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=False)

    if info is None:
        print(f"Warning: could not extract info from: {url}")
        return []

    if "entries" in info:
        return [
            VideoEntry(
                url=entry.get("url") or f"https://www.youtube.com/watch?v={entry['id']}",
                title=entry.get("title") or entry.get("id") or "Unknown",
            )
            for entry in info["entries"]
            if entry is not None
        ]

    return [VideoEntry(url=url, title=info.get("title") or url)]


def download_single_video(
    entry: VideoEntry,
    progress: DownloadProgress,
    output_dir: Path,
    result: DownloadResult,
) -> None:
    import yt_dlp

    index: int = progress.get_next_index()
    print(f"[{index}/{progress.total}] ▶  {entry.title}")

    options: dict = build_single_download_options(output_dir)

    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            error_code: int = downloader.download([entry.url])
        if error_code == 0:
            result.record_success()
            print(f"[{index}/{progress.total}] ✓  {entry.title}")
        else:
            result.record_failure()
            print(f"[{index}/{progress.total}] ✗  {entry.title}")
    except Exception:
        result.record_failure()
        print(f"[{index}/{progress.total}] ✗  {entry.title}")


def download_all_concurrently(entries: list[VideoEntry], output_dir: Path) -> DownloadResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = DownloadResult()
    progress = DownloadProgress(total=len(entries))

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_DOWNLOADS) as executor:
        futures = [
            executor.submit(download_single_video, entry, progress, output_dir, result)
            for entry in entries
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                pass

    return result


def collect_urls() -> list[str]:
    if len(sys.argv) > 1:
        return sys.argv[1:]
    raw: str = input("Enter YouTube URL(s) separated by spaces: ").strip()
    return raw.split() if raw else []


def main() -> None:
    ensure_dependencies()

    urls: list[str] = collect_urls()
    if not urls:
        print("Error: no URLs provided.")
        sys.exit(1)

    start_time: float = time.perf_counter()

    print("Extracting video list...")
    all_entries: list[VideoEntry] = []
    for url in urls:
        entries: list[VideoEntry] = extract_entries_from_url(url)
        all_entries.extend(entries)

    if not all_entries:
        print("Error: no videos found.")
        sys.exit(1)

    print(f"Found {len(all_entries)} video(s). Starting {MAX_CONCURRENT_DOWNLOADS} parallel downloads...\n")

    result: DownloadResult = download_all_concurrently(all_entries, OUTPUT_DIR)
    result.print_summary()

    elapsed: float = time.perf_counter() - start_time
    print(f"Total time: {format_duration(elapsed)}")


if __name__ == "__main__":
    main()