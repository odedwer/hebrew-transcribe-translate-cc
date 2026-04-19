# heb-transcribe Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the monolithic main.py into focused modules and add single-file, sequential-directory, and SLURM-batch run modes via a proper CLI.

**Architecture:** The code is split into four modules (transcriber, translator, exporter, main) plus a shared models module. main.py handles CLI dispatch; each other module has one responsibility and communicates via a shared Segment dataclass defined in models.py.

**Tech Stack:** Python 3.12, stable-ts (stable_whisper), faster-whisper, ollama, moviepy, torch, pathlib, argparse, logging, pytest

---

### Task 1: Create models.py with Segment dataclass

**Files:**
- Create: `models.py`
- Create: `tests/__init__.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
from models import Segment

def test_segment_creation():
    seg = Segment(start=0.0, end=1.5, text="Hello")
    assert seg.start == 0.0
    assert seg.end == 1.5
    assert seg.text == "Hello"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models'`

- [ ] **Step 3: Write minimal implementation**

```python
# models.py
from dataclasses import dataclass


@dataclass
class Segment:
    start: float
    end: float
    text: str
```

Also create `tests/__init__.py` as an empty file.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py -v`
Expected: 1 PASS

- [ ] **Step 5: Commit**

```bash
git add models.py tests/__init__.py tests/test_models.py
git commit -m "feat: add Segment dataclass"
```

---

### Task 2: Create exporter.py

**Files:**
- Create: `exporter.py`
- Create: `tests/test_exporter.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_exporter.py
import tempfile
from pathlib import Path
from exporter import format_timestamp, write_srt
from models import Segment


def test_format_timestamp_zero():
    assert format_timestamp(0.0) == "00:00:00,000"


def test_format_timestamp_all_units():
    assert format_timestamp(3723.5) == "01:02:03,500"


def test_format_timestamp_millis():
    assert format_timestamp(1.25) == "00:00:01,250"


def test_write_srt_creates_file():
    segments = [
        Segment(start=0.0, end=1.5, text="Hello world"),
        Segment(start=2.0, end=3.5, text="Second line"),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "output.srt"
        write_srt(segments, out)
        content = out.read_text(encoding="utf-8")
    assert "1\n" in content
    assert "00:00:00,000 --> 00:00:01,500\n" in content
    assert "Hello world\n" in content
    assert "2\n" in content
    assert "00:00:02,000 --> 00:00:03,500\n" in content
    assert "Second line\n" in content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_exporter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'exporter'`

- [ ] **Step 3: Write implementation**

```python
# exporter.py
import datetime
from pathlib import Path
from models import Segment


def format_timestamp(seconds: float) -> str:
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def write_srt(segments: list[Segment], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(
                f"{i}\n"
                f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n"
                f"{seg.text}\n\n"
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_exporter.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add exporter.py tests/test_exporter.py
git commit -m "feat: add exporter module"
```

---

### Task 3: Create translator.py

**Files:**
- Create: `translator.py`
- Create: `tests/test_translator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_translator.py
from unittest.mock import patch
from translator import translate, _parse_response
from models import Segment


def test_parse_response_basic():
    raw = "REF_0: Hello\nREF_1: World\nREF_2: Foo"
    result = _parse_response(raw, count=3)
    assert result == ["Hello", "World", "Foo"]


def test_parse_response_missing_index_falls_back():
    raw = "REF_0: Hello\nREF_2: Foo"
    fallback = ["orig0", "orig1", "orig2"]
    result = _parse_response(raw, count=3, fallback=fallback)
    assert result[1] == "orig1"


def test_translate_returns_segments_with_translated_text():
    segments = [
        Segment(start=0.0, end=1.0, text="שלום"),
        Segment(start=1.0, end=2.0, text="עולם"),
    ]
    mock_response = {"response": "REF_0: Hello\nREF_1: World"}
    with patch("translator.ollama.generate", return_value=mock_response):
        result = translate(segments, model="test-model", num_blocks=1)
    assert len(result) == 2
    assert result[0].text == "Hello"
    assert result[1].text == "World"
    assert result[0].start == 0.0
    assert result[1].end == 2.0


def test_translate_merges_short_segments_into_previous():
    segments = [
        Segment(start=0.0, end=1.0, text="שלום"),   # 1.0s — kept
        Segment(start=1.0, end=1.2, text="ע"),       # 0.2s — short, merged into previous
        Segment(start=1.5, end=3.0, text="עולם"),   # 1.5s — kept
    ]
    mock_response = {"response": "REF_0: Hello\nREF_1: W\nREF_2: World"}
    with patch("translator.ollama.generate", return_value=mock_response):
        result = translate(segments, model="test-model", num_blocks=1, min_duration=0.4)
    assert len(result) == 2
    assert result[0].end == 1.2
    assert result[1].text == "World"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_translator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'translator'`

- [ ] **Step 3: Write implementation**

```python
# translator.py
import logging
import ollama
from models import Segment

logger = logging.getLogger(__name__)

MIN_DURATION = 0.4


def _parse_response(raw: str, count: int, fallback: list[str] | None = None) -> list[str]:
    translations: dict[int, str] = {}
    for line in raw.strip().split("\n"):
        if "REF_" in line and ":" in line:
            try:
                parts = line.split(":", 1)
                idx = int("".join(filter(str.isdigit, parts[0])))
                translations[idx] = parts[1].strip()
            except (ValueError, IndexError):
                continue
    if fallback is None:
        fallback = [""] * count
    return [translations.get(i, fallback[i]) for i in range(count)]


def _translate_block(segments: list[Segment], block_num: int, total_blocks: int, model: str) -> list[str]:
    if not segments:
        return []
    numbered = "\n".join(f"REF_{i}: {seg.text.strip()}" for i, seg in enumerate(segments))
    prompt = (
        f"You are a professional translator for university lectures. Translating Block {block_num}/{total_blocks}.\n"
        "TASK: Translate the following Hebrew segments into natural, easy-to-understand English.\n\n"
        "REQUIREMENTS:\n"
        "1. Direct Mapping: One translation per REF_ID. Do not skip or merge lines.\n"
        "2. Tone: Common English. Clear and natural (not overly formal).\n"
        "3. Cleanup: If the speaker stutters or repeats a fragment, translate only the final intended thought.\n"
        "4. Grammatical: Keep all connecting words (of, and, the).\n"
        "5. Output Format: 'REF_x: [Translation]'.\n\n"
        f"DATA:\n{numbered}"
    )
    try:
        response = ollama.generate(model=model, prompt=prompt, options={"temperature": 0})
        return _parse_response(response["response"], count=len(segments), fallback=[s.text for s in segments])
    except Exception:
        logger.exception("Translation failed for block %d", block_num)
        return [s.text for s in segments]


def translate(
    segments: list[Segment],
    model: str,
    num_blocks: int,
    min_duration: float = MIN_DURATION,
) -> list[Segment]:
    block_size = (len(segments) // num_blocks) + 1
    blocks = [segments[i: i + block_size] for i in range(0, len(segments), block_size)]
    result: list[Segment] = []

    for i, block in enumerate(blocks):
        translations = _translate_block(block, i + 1, len(blocks), model)
        for j, text in enumerate(translations):
            if j >= len(block):
                break
            seg = block[j]
            if (seg.end - seg.start) > min_duration:
                result.append(Segment(start=seg.start, end=seg.end, text=text))
            elif result:
                result[-1] = Segment(start=result[-1].start, end=seg.end, text=result[-1].text)

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_translator.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add translator.py tests/test_translator.py
git commit -m "feat: add translator module"
```

---

### Task 4: Create transcriber.py

**Files:**
- Create: `transcriber.py`
- Create: `tests/test_transcriber.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_transcriber.py
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
from transcriber import transcribe
from models import Segment


def _mock_seg(start, end, text):
    seg = MagicMock()
    seg.start = start
    seg.end = end
    seg.text = text
    return seg


def test_transcribe_loads_checkpoint_if_exists():
    mock_seg = _mock_seg(0.0, 1.0, "שלום")
    mock_result = MagicMock()
    mock_result.segments = [mock_seg]

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = Path(tmpdir) / "checkpoint.json"
        checkpoint.write_text("{}")
        with patch("transcriber.stable_whisper.WhisperResult", return_value=mock_result) as mock_wr, \
             patch("transcriber.stable_whisper.load_faster_whisper") as mock_load:
            result = transcribe(Path(tmpdir) / "video.mp4", checkpoint, device="cpu")
            mock_wr.assert_called_once_with(str(checkpoint))
            mock_load.assert_not_called()

    assert len(result) == 1
    assert isinstance(result[0], Segment)
    assert result[0].text == "שלום"


def test_transcribe_runs_model_if_no_checkpoint():
    mock_seg = _mock_seg(0.0, 1.0, "שלום")
    mock_result = MagicMock()
    mock_result.segments = [mock_seg]
    mock_model = MagicMock()
    mock_model.transcribe.return_value = mock_result

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = Path(tmpdir) / "checkpoint.json"
        video = Path(tmpdir) / "video.mp4"
        video.write_bytes(b"fake")

        with patch("transcriber.stable_whisper.load_faster_whisper", return_value=mock_model), \
             patch("transcriber.VideoFileClip") as mock_clip_cls, \
             patch("transcriber.torch.cuda.empty_cache"):
            mock_clip = MagicMock()
            mock_clip_cls.return_value = mock_clip
            result = transcribe(video, checkpoint, device="cpu", whisper_model="test-model")
            mock_model.transcribe.assert_called_once()

    assert len(result) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_transcriber.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'transcriber'`

- [ ] **Step 3: Write implementation**

```python
# transcriber.py
import logging
from pathlib import Path

import torch
import stable_whisper
from moviepy import VideoFileClip

from models import Segment

logger = logging.getLogger(__name__)

WHISPER_MODEL = "ivrit-ai/faster-whisper-v2-d4"
MERGE_GAP = 0.4
MAX_CHARS = 55
MAX_DURATION = 5.0


def transcribe(
    video_path: Path,
    checkpoint_path: Path,
    device: str,
    whisper_model: str = WHISPER_MODEL,
) -> list[Segment]:
    audio_temp = video_path.with_suffix(".tmp.mp3")

    if checkpoint_path.exists():
        logger.info("Loading transcription checkpoint: %s", checkpoint_path)
        result = stable_whisper.WhisperResult(str(checkpoint_path))
    else:
        logger.info("Extracting audio from %s", video_path)
        video = VideoFileClip(str(video_path))
        video.audio.write_audiofile(str(audio_temp), logger=None, fps=16000)
        video.close()

        logger.info("Transcribing on %s", device)
        model = stable_whisper.load_faster_whisper(whisper_model, device=device)
        result = model.transcribe(str(audio_temp), language="he", vad=True, regroup=True)
        result.save_as_json(str(checkpoint_path))
        del model
        torch.cuda.empty_cache()

    if audio_temp.exists():
        audio_temp.unlink()

    logger.info("Refining segments")
    result.merge_by_gap(MERGE_GAP)
    result.split_by_length(max_chars=MAX_CHARS)
    result.clamp_max(MAX_DURATION)

    return [Segment(start=s.start, end=s.end, text=s.text) for s in result.segments]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_transcriber.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add transcriber.py tests/test_transcriber.py
git commit -m "feat: add transcriber module"
```

---

### Task 5: Rewrite main.py with CLI and all run modes

**Files:**
- Modify: `main.py` (full rewrite)
- Create: `tests/test_main.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_main.py
import tempfile
from pathlib import Path
from main import build_parser, get_output_path, generate_sbatch_script


def test_parser_single_file_defaults():
    parser = build_parser()
    args = parser.parse_args(["video.mp4"])
    assert args.input == Path("video.mp4")
    assert args.slurm is False
    assert args.partition == "salmon"
    assert args.gpus == 1
    assert args.time == "02:00:00"
    assert args.mem == "32G"


def test_parser_slurm_with_custom_partition():
    parser = build_parser()
    args = parser.parse_args(["mydir", "--slurm", "--partition", "gpu"])
    assert args.slurm is True
    assert args.partition == "gpu"


def test_parser_custom_output_dir():
    parser = build_parser()
    args = parser.parse_args(["video.mp4", "--output-dir", "/tmp/out"])
    assert args.output_dir == Path("/tmp/out")


def test_get_output_path_default():
    with tempfile.TemporaryDirectory() as tmpdir:
        video = Path(tmpdir) / "lecture.mp4"
        result = get_output_path(video, output_dir=None)
        assert result == Path(tmpdir) / "lecture.srt"


def test_get_output_path_custom_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        video = Path("/some/path/lecture.mp4")
        result = get_output_path(video, output_dir=Path(tmpdir))
        assert result == Path(tmpdir) / "lecture.srt"


def test_generate_sbatch_script_contains_required_directives():
    script = generate_sbatch_script(
        video_path=Path("/data/lecture.mp4"),
        partition="salmon",
        gpus=1,
        time="02:00:00",
        mem="32G",
        python_exe="python",
        script_path=Path("main.py"),
    )
    assert "#SBATCH --partition=salmon" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --time=02:00:00" in script
    assert "#SBATCH --mem=32G" in script
    assert "python main.py" in script
    assert "/data/lecture.mp4" in script
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_main.py -v`
Expected: FAIL (ImportError — old main.py has no `build_parser`, `get_output_path`, or `generate_sbatch_script`)

- [ ] **Step 3: Replace entire contents of main.py**

```python
# main.py
import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import torch

from exporter import write_srt
from models import Segment  # noqa: F401 — re-exported for convenience
from transcriber import transcribe
from translator import translate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

WHISPER_MODEL = "ivrit-ai/faster-whisper-v2-d4"
OLLAMA_MODEL = "gemma4:e4b"
NUM_BLOCKS = 8
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Transcribe and translate Hebrew video files to SRT subtitles."
    )
    p.add_argument("input", type=Path, help="Video file or directory of video files")
    p.add_argument(
        "--output-dir", "-o", type=Path, default=None,
        help="Output directory (default: same as input)",
    )
    p.add_argument("--slurm", action="store_true", help="Submit each file as a SLURM sbatch job")
    p.add_argument("--partition", default="salmon", help="SLURM partition (default: salmon)")
    p.add_argument("--gpus", type=int, default=1, help="GPUs per SLURM job (default: 1)")
    p.add_argument("--time", default="02:00:00", help="SLURM time limit (default: 02:00:00)")
    p.add_argument("--mem", default="32G", help="SLURM memory per job (default: 32G)")
    return p


def get_output_path(video_path: Path, output_dir: Path | None) -> Path:
    base = output_dir if output_dir else video_path.parent
    return base / (video_path.stem + ".srt")


def generate_sbatch_script(
    video_path: Path,
    partition: str,
    gpus: int,
    time: str,
    mem: str,
    python_exe: str,
    script_path: Path,
) -> str:
    return (
        "#!/bin/bash\n"
        f"#SBATCH --partition={partition}\n"
        f"#SBATCH --gres=gpu:{gpus}\n"
        f"#SBATCH --time={time}\n"
        f"#SBATCH --mem={mem}\n"
        f"#SBATCH --job-name=heb-transcribe-{video_path.stem}\n"
        f"#SBATCH --output=slurm_jobs/{video_path.stem}_%j.log\n"
        "\n"
        f"{python_exe} {script_path} {video_path}\n"
    )


def process_file(video_path: Path, output_dir: Path | None) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = video_path.with_name(video_path.stem + "_checkpoint.json")
    output = get_output_path(video_path, output_dir)

    logger.info("Processing %s", video_path)
    segments = transcribe(video_path, checkpoint, device, whisper_model=WHISPER_MODEL)
    segments = translate(segments, model=OLLAMA_MODEL, num_blocks=NUM_BLOCKS)
    write_srt(segments, output)
    logger.info("Saved %s", output)


def process_directory_sequential(directory: Path, output_dir: Path | None) -> None:
    videos = sorted(p for p in directory.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        logger.warning("No video files found in %s", directory)
        return
    for video in videos:
        process_file(video, output_dir)


def process_directory_slurm(
    directory: Path,
    output_dir: Path | None,
    partition: str,
    gpus: int,
    time: str,
    mem: str,
) -> None:
    videos = sorted(p for p in directory.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        logger.warning("No video files found in %s", directory)
        return

    jobs_dir = directory / "slurm_jobs"
    jobs_dir.mkdir(exist_ok=True)

    python_exe = sys.executable
    script_path = Path(__file__).resolve()

    for video in videos:
        script = generate_sbatch_script(
            video_path=video,
            partition=partition,
            gpus=gpus,
            time=time,
            mem=mem,
            python_exe=python_exe,
            script_path=script_path,
        )
        sbatch_file = jobs_dir / f"{video.stem}.sh"
        sbatch_file.write_text(script)
        result = subprocess.run(["sbatch", str(sbatch_file)], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("Submitted %s: %s", video.name, result.stdout.strip())
        else:
            logger.error("Failed to submit %s: %s", video.name, result.stderr.strip())


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    input_path: Path = args.input

    if input_path.is_file():
        process_file(input_path, args.output_dir)
    elif input_path.is_dir():
        if args.slurm:
            process_directory_slurm(
                input_path,
                args.output_dir,
                partition=args.partition,
                gpus=args.gpus,
                time=args.time,
                mem=args.mem,
            )
        else:
            process_directory_sequential(input_path, args.output_dir)
    else:
        logger.error("Input path does not exist: %s", input_path)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_main.py -v`
Expected: 6 PASS

- [ ] **Step 5: Run all tests**

Run: `pytest -v`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: rewrite main.py with CLI and all run modes"
```

---

### Task 6: Write README.md

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README.md**

```markdown
# heb-transcribe

Transcribes Hebrew video files and translates subtitles to English, producing `.srt` files.
Uses [faster-whisper](https://github.com/guillaumekientz/faster-whisper) for transcription
and [Ollama](https://ollama.com) for translation.

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- [Ollama](https://ollama.com) running locally with the `gemma4:e4b` model pulled
- CUDA-capable GPU (recommended; falls back to CPU)
- `ffmpeg` on PATH

## Installation

```bash
uv sync
```

## Usage

### Single file

```bash
python main.py lecture.mp4
# Output: lecture.srt alongside the input file

python main.py lecture.mp4 --output-dir /path/to/output
```

### Directory — sequential

```bash
python main.py /path/to/videos/
# Processes all .mp4 / .mkv / .avi / .mov files in order
```

### Directory — SLURM batch

```bash
python main.py /path/to/videos/ --slurm

# With custom SLURM parameters:
python main.py /path/to/videos/ --slurm \
  --partition salmon \
  --gpus 1 \
  --time 04:00:00 \
  --mem 64G
```

Job scripts are written to `{input_dir}/slurm_jobs/` and logs to
`{input_dir}/slurm_jobs/{stem}_{job_id}.log`.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir`, `-o` | same as input | Directory for `.srt` output files |
| `--slurm` | off | Submit directory jobs via `sbatch` |
| `--partition` | `salmon` | SLURM partition |
| `--gpus` | `1` | GPUs per job |
| `--time` | `02:00:00` | SLURM time limit |
| `--mem` | `32G` | SLURM memory per job |

## Checkpoints

Transcription results are cached as `{stem}_checkpoint.json` next to the input file.
Delete this file to force re-transcription.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README"
```

---

### Task 7: Final verification

- [ ] **Step 1: Run full test suite**

Run: `pytest -v`
Expected: all tests pass (9 total across models, exporter, translator, transcriber, main)

- [ ] **Step 2: Verify CLI help**

Run: `python main.py --help`
Expected: help text listing `input`, `--output-dir`, `--slurm`, `--partition`, `--gpus`, `--time`, `--mem`

- [ ] **Step 3: Commit any remaining untracked files**

```bash
git status
git add pyproject.toml .gitignore .python-version
git commit -m "chore: include project config files"
```
