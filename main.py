import os
import json
import torch
import ollama
import datetime
import stable_whisper
from tqdm import tqdm
from moviepy import VideoFileClip

# 1. Environment & Setup
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_FILE = "transcription_checkpoint.json"
OLLAMA_MODEL = "gemma4:e4b"

def format_timestamp(seconds: float) -> str:
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def translate_large_block(segments, block_num, total_blocks):
    """Translates a massive chunk of text for maximum coherence."""
    if not segments: return []
    
    numbered_text = "\n".join([f"REF_{i}: {seg.text.strip()}" for i, seg in enumerate(segments)])

    prompt = (
        f"You are a professional translator for university lectures. Translating Block {block_num}/{total_blocks}.\n"
        "TASK: Translate the following Hebrew segments into natural, easy-to-understand English.\n\n"
        "REQUIREMENTS:\n"
        "1. Direct Mapping: One translation per REF_ID. Do not skip or merge lines.\n"
        "2. Tone: Common English. Clear and natural (not overly formal).\n"
        "3. Cleanup: If the speaker stutters or repeats a fragment, translate only the final intended thought.\n"
        "4. Grammatical: Keep all connecting words (of, and, the).\n"
        "5. Output Format: 'REF_x: [Translation]'.\n\n"
        f"DATA:\n{numbered_text}"
    )

    try:
        response = ollama.generate(model=OLLAMA_MODEL, prompt=prompt, options={'temperature': 0})
        raw_lines = response['response'].strip().split('\n')
        
        translations = {}
        for line in raw_lines:
            if "REF_" in line and ":" in line:
                try:
                    parts = line.split(":", 1)
                    idx = int(''.join(filter(str.isdigit, parts[0])))
                    translations[idx] = parts[1].strip()
                except: continue
        
        return [translations.get(i, segments[i].text) for i in range(len(segments))]
    except Exception as e:
        return [s.text for s in segments]

def main(video_input, subtitle_output):
    audio_temp = "temp_audio_final.mp3"

    # --- 1. TRANSCRIPTION ---
    if os.path.exists(CHECKPOINT_FILE):
        print("--- Loading Transcription Cache ---")
        result = stable_whisper.WhisperResult(CHECKPOINT_FILE)
    else:
        video = VideoFileClip(video_input)
        video.audio.write_audiofile(audio_temp, logger=None, fps=16000)
        video.close()
        
        print(f"--- Transcribing on {DEVICE} ---")
        model = stable_whisper.load_faster_whisper("ivrit-ai/faster-whisper-v2-d4", device=DEVICE)
        result = model.transcribe(audio_temp, language='he', vad=True, regroup=True)
        result.save_as_json(CHECKPOINT_FILE)
        
        del model
        torch.cuda.empty_cache()

    # --- 2. REFINEMENT ---
    print("--- Refining Segments ---")
    result.merge_by_gap(0.4) 
    result.split_by_length(max_chars=55) 
    result.clamp_max(5.0) 

    all_segments = list(result.segments)
    
    # 8 Large blocks total
    num_blocks = 8
    block_size = (len(all_segments) // num_blocks) + 1
    
    final_processed = []

    # --- 3. BATCH TRANSLATION WITH TQDM ---
    print(f"--- Translating {len(all_segments)} segments in {num_blocks} blocks ---")
    
    # Create the block list beforehand for the progress bar
    blocks = [all_segments[i : i + block_size] for i in range(0, len(all_segments), block_size)]
    
    for i, current_batch in enumerate(tqdm(blocks, desc="Ollama Translation", unit="block")):
        block_idx = i + 1
        translations = translate_large_block(current_batch, block_idx, len(blocks))
        
        for j, trans in enumerate(translations):
            if j < len(current_batch):
                if (current_batch[j].end - current_batch[j].start) > 0.4:
                    final_processed.append({
                        'start': current_batch[j].start,
                        'end': current_batch[j].end,
                        'text': trans
                    })
                elif final_processed:
                    final_processed[-1]['end'] = current_batch[j].end

    # --- 4. EXPORT ---
    print(f"--- Exporting {subtitle_output} ---")
    with open(subtitle_output, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(final_processed, start=1):
            f.write(f"{i}\n{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n{seg['text']}\n\n")

    if os.path.exists(audio_temp):
        os.remove(audio_temp)
    print("\n--- Process Finished Successfully ---")

if __name__ == "__main__":
    main("Geometric Intuitions - Part I_default.mp4", "lecture_subtitles.srt")