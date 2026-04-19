# heb-transcribe Refactor Design
Date: 2026-04-19

## Overview

Refactor `main.py` into focused modules and add three run modes: single file, sequential directory, and SLURM batch directory.

## Modules

| File | Responsibility |
|------|----------------|
| `transcriber.py` | Extract audio from video, run faster-whisper, save/load per-file checkpoint, refine segments |
| `translator.py` | Batch-translate Hebrew segments via ollama, parse REF responses |
| `exporter.py` | Format timestamps, write SRT file |
| `main.py` | CLI entry point (argparse), dispatch to single/sequential/slurm modes |

## Data Flow

```
main.py
  └─ transcriber.transcribe(video_path, checkpoint_path) -> List[Segment]
  └─ translator.translate(segments) -> List[TranslatedSegment]
  └─ exporter.write_srt(segments, output_path)
```

## CLI Interface

```
python main.py <input> [options]

Arguments:
  input                 Video file or directory of video files

Options:
  --output-dir, -o      Output directory (default: same as input)
  --slurm               Submit each file as a SLURM sbatch job
  --partition           SLURM partition (default: salmon)
  --gpus                GPU count for SLURM jobs (default: 1)
  --time                SLURM time limit (default: 02:00:00)
  --mem                 SLURM memory (default: 32G)
```

## Run Modes

### Single file
`input` is a file path. Transcribe, translate, export SRT to `output-dir` (default: same dir).

### Directory sequential
`input` is a directory. Scan for `.mp4`, `.mkv`, `.avi`, `.mov` files. Process each sequentially, one after another.

### Directory SLURM
`input` is a directory + `--slurm` flag. For each video file, write a sbatch script to `{input_dir}/slurm_jobs/` and submit with `sbatch`. Each job calls `python main.py <single_file>`.

## Checkpoints

Per-file checkpoint: `{input_stem}_checkpoint.json` in the same directory as the input file. Allows resuming transcription without rerunning whisper.

## Config Constants (main.py top-level)

- `WHISPER_MODEL = "ivrit-ai/faster-whisper-v2-d4"`
- `OLLAMA_MODEL = "gemma4:e4b"`
- `NUM_BLOCKS = 8`
- `MERGE_GAP = 0.4`
- `MAX_CHARS = 55`
- `MAX_DURATION = 5.0`
- `MIN_SEGMENT_DURATION = 0.4`
- `VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov"}`

## Best Practices Applied

- `logging` module replaces all `print` statements
- Type hints throughout
- No hardcoded file paths
- `pathlib.Path` for all path operations
- Clean separation of concerns between modules

## README

Generated `README.md` covering: project description, requirements, installation, usage for all three modes, and SLURM notes.
