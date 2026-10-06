#!/usr/bin/env python3
"""
Stage 1 spike v2 — one-shot diagnose + run.

Why v2: v1 crashed inside faster-whisper's INTERNAL audio decoder
(av.open(..., metadata_errors=...) rejected by the installed PyAV).
This version decodes the audio ITSELF and hands the model a numpy array —
which is exactly what our Stage 1 design decided ("own the decode, don't
delegate"), so it sidesteps that whole class of library-version breakage.

It prints an environment report, tries several decode paths and tells you which
one worked, then runs the transcription and reports timing + the rich output
(word timestamps + confidence) that our output contract needs.

USAGE
  python stage1_spike_v2.py --audio jfk.flac --model large-v3-turbo
  python stage1_spike_v2.py --audio jfk.flac --model large-v3-turbo --backend mlx \
      --model mlx-community/whisper-large-v3-turbo
"""
import argparse, os, platform, subprocess, sys, time

SR = 16000


# ---------------------------------------------------------------- environment
def env_report():
    print("=" * 60)
    print("ENVIRONMENT")
    print("=" * 60)
    print(f"python      : {sys.version.split()[0]}  ({platform.machine()})")
    print(f"platform    : {platform.platform()}")
    try:
        mem = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        print(f"RAM         : {int(mem)/(1024**3):.1f} GB")
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        print(f"cpu         : {cpu}")
    except Exception:
        pass
    for mod in ("faster_whisper", "av", "ctranslate2", "mlx_whisper", "numpy"):
        try:
            m = __import__(mod)
            print(f"{mod:<12}: {getattr(m, '__version__', '?')}")
        except Exception as e:
            print(f"{mod:<12}: NOT INSTALLED ({type(e).__name__})")
    print(f"ffmpeg      : {'yes' if _which('ffmpeg') else 'NO (brew install ffmpeg)'}")
    print()


def _which(prog):
    from shutil import which
    return which(prog)


# -------------------------------------------------------------------- decoding
def decode_library(path):
    """Path A: let faster-whisper do it (the one that failed in v1)."""
    from faster_whisper.audio import decode_audio
    return decode_audio(path, sampling_rate=SR)


def decode_ffmpeg(path):
    """Path B: shell out to ffmpeg. Most robust if ffmpeg is installed."""
    import numpy as np
    if not _which("ffmpeg"):
        raise RuntimeError("ffmpeg not on PATH")
    cmd = ["ffmpeg", "-nostdin", "-threads", "0", "-i", path,
           "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(SR), "-"]
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True).stdout
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0


def decode_av(path):
    """Path C: PyAV directly, WITHOUT the metadata_errors kwarg that broke v1."""
    import av, numpy as np
    with av.open(path, mode="r") as container:
        stream = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=SR)
        chunks = []

        def _emit(frame):
            res = resampler.resample(frame)
            if res is None:
                return
            if not isinstance(res, list):      # API differs across PyAV versions
                res = [res]
            for rf in res:
                chunks.append(rf.to_ndarray().reshape(-1))

        for frame in container.decode(stream):
            _emit(frame)
        _emit(None)                            # flush
    if not chunks:
        raise RuntimeError("no audio frames decoded")
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def decode(path):
    print("=" * 60)
    print("DECODE (trying paths in order)")
    print("=" * 60)
    for name, fn in (("faster-whisper internal", decode_library),
                     ("ffmpeg subprocess", decode_ffmpeg),
                     ("PyAV direct", decode_av)):
        try:
            t = time.time()
            audio = fn(path)
            print(f"  [OK]   {name}: {len(audio)} samples "
                  f"({len(audio)/SR:.1f}s) in {time.time()-t:.2f}s")
            print(f"  --> using: {name}\n")
            return audio
        except Exception as e:
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
    raise SystemExit("\nAll decode paths failed. Install ffmpeg: brew install ffmpeg")


# ---------------------------------------------------------------- transcription
def run_faster(model_name, audio):
    from faster_whisper import WhisperModel
    print("=" * 60)
    print(f"TRANSCRIBE — faster-whisper / {model_name}")
    print("=" * 60)
    t0 = time.time()
    model = WhisperModel(model_name, device="auto", compute_type="auto")
    load_s = time.time() - t0
    print(f"[load] {load_s:.1f}s  (should be seconds now that weights are cached)")

    audio_s = len(audio) / SR
    t1 = time.time()
    segments, info = model.transcribe(audio, language="en", beam_size=5,
                                      word_timestamps=True, vad_filter=True)
    segs = list(segments)                      # generator — force the work
    dt = time.time() - t1

    print(f"[transcribe] audio={audio_s:.1f}s  proc={dt:.1f}s  "
          f"RTF={dt/audio_s:.2f}  speed={audio_s/dt:.1f}x realtime")
    print(f"[text] {' '.join(s.text.strip() for s in segs)}")
    if segs:
        s0 = segs[0]
        print(f"[segments] {len(segs)}  avg_logprob={s0.avg_logprob:.3f}  "
              f"no_speech_prob={s0.no_speech_prob:.3f}")
        if s0.words:
            w = s0.words[0]
            print(f"[word0] '{w.word.strip()}' {w.start:.2f}-{w.end:.2f}s "
                  f"prob={w.probability:.3f}")
    print("\n[contract check] word timestamps + confidence present: "
          f"{bool(segs and segs[0].words)}")


def run_mlx(model_repo, audio):
    import mlx_whisper
    print("=" * 60)
    print(f"TRANSCRIBE — mlx-whisper / {model_repo}")
    print("=" * 60)
    audio_s = len(audio) / SR
    t1 = time.time()
    res = mlx_whisper.transcribe(audio, path_or_hf_repo=model_repo,
                                 language="en", word_timestamps=True)
    dt = time.time() - t1
    print(f"[load+transcribe] audio={audio_s:.1f}s  total={dt:.1f}s  "
          f"RTF={dt/audio_s:.2f}  speed={audio_s/dt:.1f}x realtime")
    print(f"[text] {res['text'].strip()}")
    segs = res.get("segments") or []
    if segs:
        print(f"[segments] {len(segs)}  no_speech_prob={segs[0].get('no_speech_prob')}")
        if segs[0].get("words"):
            w = segs[0]["words"][0]
            print(f"[word0] '{w['word'].strip()}' {w['start']:.2f}-{w['end']:.2f}s "
                  f"prob={w.get('probability')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--backend", choices=["faster", "mlx"], default="faster")
    ap.add_argument("--repeat", type=int, default=1,
                    help="tile the decoded audio N times to simulate a longer "
                         "meeting. Short clips are overhead-dominated, so RTF on "
                         "an 11s file is NOT representative; use --repeat 20+ for "
                         "a realistic long-form number.")
    args = ap.parse_args()

    if not os.path.exists(args.audio):
        raise SystemExit(f"audio file not found: {args.audio}")

    env_report()
    audio = decode(args.audio)
    if args.repeat > 1:
        import numpy as np
        audio = np.tile(audio, args.repeat)
        print(f"[repeat] tiled x{args.repeat} -> {len(audio)/SR/60:.1f} min of audio\n")
    if args.backend == "faster":
        run_faster(args.model, audio)
    else:
        run_mlx(args.model, audio)
    print("\nDONE.")
