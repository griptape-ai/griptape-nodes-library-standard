"""FFmpeg utility functions for cross-platform executable path resolution."""

import subprocess
from collections.abc import Callable

import static_ffmpeg.run  # type: ignore[import-untyped]


def get_ffmpeg_path() -> str:
    """Get the path to ffmpeg executable using static_ffmpeg for cross-platform compatibility.

    This function handles finding the ffmpeg executable across different platforms:
    - Windows: Uses static-ffmpeg's platform-specific resolution
    - Linux: Uses static-ffmpeg's platform-specific resolution
    - macOS: Uses static-ffmpeg's platform-specific resolution

    Returns:
        Path to the ffmpeg executable

    Raises:
        RuntimeError: If ffmpeg is not found or static-ffmpeg is not properly installed
    """
    # FAILURE CASES FIRST
    try:
        ffmpeg_path, _ = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
    except (FileNotFoundError, OSError, ImportError) as e:
        error_msg = f"FFmpeg not found. Please ensure static-ffmpeg is properly installed. Error: {e!s}"
        raise RuntimeError(error_msg) from e

    # SUCCESS PATH AT END
    return ffmpeg_path


def get_ffprobe_path() -> str:
    """Get the path to ffprobe executable using static_ffmpeg for cross-platform compatibility.

    This function handles finding the ffprobe executable across different platforms:
    - Windows: Uses static-ffmpeg's platform-specific resolution
    - Linux: Uses static-ffmpeg's platform-specific resolution
    - macOS: Uses static-ffmpeg's platform-specific resolution

    Returns:
        Path to the ffprobe executable

    Raises:
        RuntimeError: If ffprobe is not found or static-ffmpeg is not properly installed
    """
    # FAILURE CASES FIRST
    try:
        _, ffprobe_path = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
    except (FileNotFoundError, OSError, ImportError) as e:
        error_msg = f"FFprobe not found. Please ensure static-ffmpeg is properly installed. Error: {e!s}"
        raise RuntimeError(error_msg) from e

    # SUCCESS PATH AT END
    return ffprobe_path


def get_ffmpeg_paths() -> tuple[str, str]:
    """Get both ffmpeg and ffprobe executable paths using static_ffmpeg.

    This is a convenience function that returns both paths in a single call,
    which is useful when you need both executables.

    Returns:
        Tuple of (ffmpeg_path, ffprobe_path)

    Raises:
        RuntimeError: If either executable is not found or static-ffmpeg is not properly installed
    """
    # FAILURE CASES FIRST
    try:
        ffmpeg_path, ffprobe_path = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
    except (FileNotFoundError, OSError, ImportError) as e:
        error_msg = f"FFmpeg/FFprobe not found. Please ensure static-ffmpeg is properly installed. Error: {e!s}"
        raise RuntimeError(error_msg) from e

    # SUCCESS PATH AT END
    return ffmpeg_path, ffprobe_path


def run_ffmpeg_cmd(
    cmd: list[str],
    *,
    log: Callable[[str], None] | None = None,
    timeout: int = 300,
) -> None:
    """Run an ffmpeg command with standard error handling and optional logging."""
    if log:
        log(f"Running ffmpeg command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)  # noqa: S603
        if result.stderr and log:
            log(f"FFmpeg stderr: {result.stderr}\n")
    except subprocess.TimeoutExpired as e:
        error_msg = f"FFmpeg timed out after {timeout}s"
        if log:
            log(f"ERROR: {error_msg}\n")
        raise ValueError(error_msg) from e
    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg error: {e.stderr}"
        if log:
            log(f"ERROR: {error_msg}\n")
        raise ValueError(error_msg) from e
