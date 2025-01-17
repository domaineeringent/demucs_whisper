import os
import psutil
import logging
import warnings
import gc
import torch
import whisper
from pydub import AudioSegment
from fastapi import FastAPI, File, UploadFile, HTTPException
from starlette.responses import JSONResponse, FileResponse
import uvicorn
import json
import zipfile
import random
import string
import threading
import soundfile as sf
import torchaudio
import ffmpeg

from demucs.pretrained import get_model
from demucs.apply import apply_model

# -------------------- CONFIGURE LOGGING -------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# -------------------- GLOBALS -------------------- #
# Enable cuDNN benchmarking and optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
app = FastAPI()

# Model caches
MODEL_CACHE = {}
WHISPER_MODEL = None

# Semaphore to limit concurrent jobs
MAX_CONCURRENT_JOBS = 2  # Adjust based on VRAM and performance
gpu_semaphore = threading.Semaphore(MAX_CONCURRENT_JOBS)

# -------------------- MODEL LOADING HELPERS -------------------- #
def get_demucs_model(model_name: str):
    """
    Load and cache Demucs model.
    """
    global MODEL_CACHE
    if model_name not in MODEL_CACHE:
        logging.info(f"Loading Demucs model: {model_name}...")
        model = get_model(model_name)
        model.to(DEVICE)
        MODEL_CACHE[model_name] = model
        logging.info(f"Demucs model '{model_name}' loaded successfully!")
    else:
        logging.info(f"Demucs model '{model_name}' already loaded.")
    return MODEL_CACHE[model_name]

def load_whisper_model():
    """
    Load and cache Whisper model.
    """
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        logging.info("Loading Whisper 'turbo' model...")
        WHISPER_MODEL = whisper.load_model("turbo", device=DEVICE)
        logging.info("Whisper model loaded successfully!")
    else:
        logging.info("Whisper model already loaded.")
    return WHISPER_MODEL

# -------------------- AUDIO PROCESSING HELPERS -------------------- #
def convert_audio_to_wav(input_path: str) -> str:
    """
    Ensure audio is in 44.1kHz stereo WAV format.
    """
    output_path = input_path + "_converted.wav"
    try:
        ffmpeg.input(input_path).output(
            output_path,
            acodec='pcm_s16le',
            ar=44100,
            ac=2
        ).overwrite_output().run(quiet=True)
        logging.info(f"Converted audio to WAV: {output_path}")
    except ffmpeg.Error as e:
        raise RuntimeError(f"FFmpeg error: {e.stderr.decode()}")
    return output_path

def preprocess_vocals_for_whisper(vocals_path: str) -> str:
    """
    **Only** the vocals stem is downsampled to mono 16kHz for Whisper transcription.
    """
    temp_path = vocals_path + '_whisper_ready.wav'
    try:
        ffmpeg.input(vocals_path).output(
            temp_path,
            ac=1,      # Mono
            ar=16000   # 16kHz
        ).overwrite_output().run(quiet=True)
        logging.info(f"Prepared vocals for Whisper: {temp_path}")
    except ffmpeg.Error as e:
        raise RuntimeError(f"FFmpeg error: {e.stderr.decode()}")
    return temp_path

def separate_stems(audio_path: str, model, shifts=1, overlap=0.25):
    """
    Separate audio into stems using Demucs, preserving **original stereo quality**.
    """
    try:
        model_name = model.name if hasattr(model, 'name') else 'htdemucs_ft'
        logging.info(f"Separating audio with {model_name} | shifts={shifts}, overlap={overlap}")

        # Load audio WITHOUT changing stereo channels
        wav, sr = torchaudio.load(audio_path)
        if sr != 44100:
            resampler = torchaudio.transforms.Resample(sr, 44100)
            wav = resampler(wav)
            sr = 44100
            logging.info(f"Resampled audio to 44100 Hz.")

        # Ensure audio remains **stereo** for Demucs
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
            logging.info("Converted mono to stereo.")
        elif wav.shape[0] > 2:
            wav = wav[:2]
            logging.info("Truncated to first two channels for stereo.")

        wav = wav / wav.abs().max()
        wav = wav.to(DEVICE)

        with torch.no_grad():
            sources = apply_model(
                model,
                wav.unsqueeze(0),  # shape: (1, channels, samples)
                shifts=shifts,
                split=True,         # Enable segmentation for long audio
                overlap=overlap,
                progress=True
            )[0]  # remove batch dim

        sources = sources.cpu()
        stems = {model.sources[i]: sources[i].numpy() for i in range(len(model.sources))}
        logging.info(f"Completed separation with {model_name}.")
        return sr, stems

    except Exception as e:
        logging.error(f"Error during stem separation with {model_name}: {e}", exc_info=True)
        return None, {}

def save_stems(stems: dict, sr: int, output_dir: str):
    """
    Save stems to WAV files.
    """
    paths = {}
    for stem_name, audio_data in stems.items():
        path = os.path.join(output_dir, f"{stem_name}.wav")
        sf.write(path, audio_data.T, sr)
        paths[stem_name] = path
        logging.info(f"Saved {stem_name} stem: {path}")
    return paths

# -------------------- WHISPER TRANSCRIPTION HELPERS -------------------- #
def transcribe_vocals(vocals_path: str):
    """
    Transcribe vocals using Whisper.
    Returns both phrases and words in the transcription.
    """
    try:
        model = load_whisper_model()
        logging.info(f"Transcribing vocals: {vocals_path}")
        result = model.transcribe(vocals_path, language="en", word_timestamps=True)

        # Build phrase segments (original phrase output from Whisper)
        phrases = []
        for seg in result.get("segments", []):
            phrases.append({
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "text": seg.get("text", "").strip()
            })

        # Build word segments with gap/duplicate handling
        words = []
        last_word = None
        last_end_time = 0.0

        for seg in result.get("segments", []):
            if words and (seg["start"] - last_end_time) > 2.0:
                logging.warning(f"Detected large gap between {last_end_time:.2f} and {seg['start']:.2f}")

            for w in seg.get("words", []):
                current_word = w.get("word", "").strip()
                start_time = w.get("start", 0.0)
                end_time = w.get("end", 0.0)

                # Deduplicate if repeated too soon
                if last_word is not None:
                    gap = start_time - last_end_time
                    if current_word == last_word and gap < 0.5:
                        logging.info(f"Skipping repeated word: {current_word} at {start_time:.3f}")
                        continue

                # Slight offset adjustments
                adjusted_start = max(0, start_time + 0.030)
                adjusted_end = end_time + 0.250

                words.append({
                    "start": adjusted_start,
                    "end": adjusted_end,
                    "text": current_word
                })

                last_word = current_word
                last_end_time = end_time

        output_json = {
            "language": result.get("language", "en"),
            "phrases": phrases,
            "words": words
        }

        logging.info("Transcription completed successfully.")
        return output_json
    except Exception as e:
        logging.error(f"Error during transcription: {e}", exc_info=True)
        return {"error": str(e)}

# -------------------- PIPELINE FUNCTION -------------------- #
def process_audio_pipeline(
    audio_file, 
    # htdemucs_ft settings
    ft_shifts=2, 
    ft_overlap=0.25
):
    """
    Master pipeline:
      1) Convert input to WAV if needed.
      2) Separate with `htdemucs_ft` => Keep all stems.
      3) Transcribe vocals with Whisper.
      4) Package all stems and transcription into a single ZIP.
    """
    if not audio_file:
        return "No file provided.", None

    ext = os.path.splitext(audio_file)[1].lower()
    if ext != ".wav":
        try:
            audio_path = convert_audio_to_wav(audio_file)
        except Exception as e:
            return f"Error converting file: {e}", None
    else:
        audio_path = audio_file
        logging.info("Input audio is already in WAV format.")

    # Create a random output folder
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    output_dir = f"processing_output_{random_suffix}"
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Created output directory: {output_dir}")

    # 1. Separate with htdemucs_ft (all stems)
    demucs_ft_model = MODEL_CACHE.get("htdemucs_ft")
    if not demucs_ft_model:
        logging.error("htdemucs_ft model not loaded.")
        return "htdemucs_ft model not loaded.", None

    logging.info("Starting htdemucs_ft separation...")
    sr_ft, stems_ft = separate_stems(audio_path, demucs_ft_model, shifts=ft_shifts, overlap=ft_overlap)

    if stems_ft:
        stems_ft_paths = save_stems(stems_ft, sr_ft, output_dir)
    else:
        return "Error during htdemucs_ft separation.", None

    # 2. Transcribe vocals with Whisper
    vocals_path = stems_ft_paths.get("vocals")
    if vocals_path and os.path.exists(vocals_path):
        # Downsample vocals to mono 16kHz for Whisper
        try:
            processed_vocals_path = preprocess_vocals_for_whisper(vocals_path)
        except Exception as e:
            logging.error(f"Error preprocessing vocals for Whisper: {e}", exc_info=True)
            return f"Error preprocessing vocals: {e}", None

        transcript = transcribe_vocals(processed_vocals_path)
        os.remove(processed_vocals_path)  # Clean up downsampled audio
    else:
        transcript = {"error": "No vocals stem found."}
        logging.warning("No vocals stem found for transcription.")

    # 3. Save separate JSON files for phrases and words
    if transcript and "error" not in transcript:
        phrases_path = os.path.join(output_dir, "phrases.json")
        words_path = os.path.join(output_dir, "words.json")
        try:
            # Save phrases.json
            with open(phrases_path, "w", encoding="utf-8") as f:
                json.dump({
                    "language": transcript.get("language", "en"),
                    "phrases": transcript.get("phrases", [])
                }, f, indent=2)
            logging.info(f"Saved phrases JSON: {phrases_path}")

            # Save words.json
            with open(words_path, "w", encoding="utf-8") as f:
                json.dump({
                    "language": transcript.get("language", "en"),
                    "words": transcript.get("words", [])
                }, f, indent=2)
            logging.info(f"Saved words JSON: {words_path}")
        except Exception as e:
            logging.error(f"Error saving separate JSON files: {e}", exc_info=True)
            return "Error saving transcription JSONs.", None
    else:
        phrases_path = None
        words_path = None
        logging.warning("Transcription failed or returned an error.")

    # 4. Package all outputs into a single ZIP
    final_zip = os.path.join(output_dir, "final_results.zip")
    try:
        with zipfile.ZipFile(final_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add Demucs_ft stems (all stems)
            for stem, path in stems_ft_paths.items():
                zipf.write(path, arcname=os.path.basename(path))
            # Add phrases.json and words.json
            if phrases_path and os.path.exists(phrases_path):
                zipf.write(phrases_path, arcname="phrases.json")
            if words_path and os.path.exists(words_path):
                zipf.write(words_path, arcname="words.json")
        logging.info(f"Created final ZIP: {final_zip}")
    except Exception as e:
        logging.error(f"Error creating final ZIP: {e}", exc_info=True)
        return "Error creating ZIP file.", None

    # -------------------- Optional Cleanup -------------------- #
    # Uncomment the following lines to remove individual stem files and JSON after zipping
    # for path in list(stems_ft_paths.values()):
    #     os.remove(path)
    # if phrases_path and os.path.exists(phrases_path):
    #     os.remove(phrases_path)
    # if words_path and os.path.exists(words_path):
    #     os.remove(words_path)

    return "Processing complete!", final_zip

# -------------------- FASTAPI ENDPOINT -------------------- #
@app.post("/api/transcribe")
async def api_transcribe(file: UploadFile = File(...)):
    """
    API endpoint for transcription.
    Upload an audio file (any supported format), get a ZIP file containing separated stems and transcription JSONs.
    """
    acquired = False  # Flag to track semaphore acquisition
    try:
        # Acquire semaphore before processing
        acquired = gpu_semaphore.acquire(blocking=False)
        if not acquired:
            logging.warning("Max concurrent jobs reached. Queuing the request.")
            raise HTTPException(status_code=429, detail="Server is busy. Please try again later.")

        # Generate a temporary filename
        temp_filename = ''.join(random.choices(string.ascii_letters + string.digits, k=12)) + "_" + file.filename
        temp_filepath = os.path.join("/tmp", temp_filename)

        # Ensure the uploads directory exists
        os.makedirs(os.path.dirname(temp_filepath), exist_ok=True)

        # Save the uploaded file to a temporary location
        with open(temp_filepath, "wb") as temp_audio:
            temp_audio.write(await file.read())

        # Define processing settings (can be parameterized as needed)
        ft_shifts = 2
        ft_overlap = 0.25

        # Process the audio pipeline
        status, final_zip = process_audio_pipeline(
            temp_filepath,
            ft_shifts=ft_shifts,
            ft_overlap=ft_overlap
        )

        if "error" in status.lower():
            logging.error(f"Processing error: {status}")
            raise HTTPException(status_code=500, detail=status)

        # Optionally, remove the temporary audio file
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
            logging.info(f"Removed temporary file: {temp_filepath}")

        # Return the ZIP file directly
        return FileResponse(
            final_zip,
            media_type='application/zip',
            filename=os.path.basename(final_zip)
        )

    except HTTPException as he:
        raise he  # Reraise HTTPExceptions to be handled by FastAPI
    except Exception as e:
        logging.error(f"API error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Ensure the semaphore is released in case of unexpected errors
        if acquired:
            gpu_semaphore.release()

# -------------------- APPLICATION STARTUP -------------------- #
@app.on_event("startup")
def startup_event():
    """
    Load models at startup to ensure they're ready for incoming requests.
    """
    logging.info("Starting up and loading models...")
    
    # Load Demucs_ft model
    try:
        get_demucs_model("htdemucs_ft")
    except Exception as e:
        logging.error(f"Failed to load htdemucs_ft model: {e}", exc_info=True)
    
    # Load Whisper model
    try:
        load_whisper_model()
    except Exception as e:
        logging.error(f"Failed to load Whisper model: {e}", exc_info=True)
    
    logging.info("All models loaded and ready.")

# -------------------- APPLICATION SHUTDOWN -------------------- #
@app.on_event("shutdown")
def shutdown_event():
    """
    Clean up resources on shutdown.
    """
    logging.info("Shutting down. Clearing GPU memory...")
    torch.cuda.empty_cache()
    gc.collect()
    logging.info("Shutdown complete.")

# -------------------- MAIN -------------------- #
if __name__ == "__main__":
    logging.info(f"Using device: {DEVICE}")

    # Launch FastAPI (Uvicorn)
    uvicorn.run(app, host="0.0.0.0", port=8000)
