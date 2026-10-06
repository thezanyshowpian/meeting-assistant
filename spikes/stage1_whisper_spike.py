#!/usr/bin/env python3
"""
Stage 1 spike — prove Whisper runs on THIS machine, free, and measure speed.

Goal: confirm the STT stage runs on the real demo hardware (native macOS /
Apple Silicon) and get a realtime factor (RTF) so we can lock model size +
runtime. Also validates our rich output contract (word-level timestamps +
per-segment confidence).

Run two backends and compare:
  faster-whisper  -> portable (CPU anywhere; our primary choice)
  mlx-whisper     -> Apple-Silicon GPU (macOS only; speed comparison)

USAGE
  # one-time setup (pick a venv/conda env you like)
  python -m venv ~/whisper_spike && source ~/whisper_spike/bin/activate
  pip install faster-whisper            # for --backend faster
  pip install mlx-whisper               # for --backend mlx  (Apple Silicon only)
  # ffmpeg: usually fine via the 'av' wheel; if audio load fails: brew install ffmpeg

  # get a test clip (or point --audio at any 1-2 min English audio you have)
  curl -L -o jfk.flac https://github.com/SYSTRAN/faster-whisper/raw/master/tests/data/jfk.flac

  # run
  python stage1_whisper_spike.py --backend faster --model large-v3-turbo --audio jfk.flac
  python stage1_whisper_spike.py --backend faster --model medium         --audio jfk.flac
  python stage1_whisper_spike.py --backend mlx    --model mlx-community/whisper-large-v3-turbo --audio jfk.flac

WHAT TO REPORT BACK
  For each run: the [load] time, the [transcribe] RTF/speed line, and whether
  the [text] looks right. That tells us which model size + runtime to lock.
"""
import argparse, time, sys


def run_faster(model_name, audio):
    from faster_whisper import WhisperModel
    print(f"[load] faster-whisper model={model_name} (device=auto, compute=auto)…", flush=True)
    t0 = time.time()
    # device='auto' uses GPU if ctranslate2 sees one; on Mac this is CPU.
    model = WhisperModel(model_name, device="auto", compute_type="auto")
    print(f"[load] done in {time.time()-t0:.1f}s", flush=True)

    t1 = time.time()
    segments, info = model.transcribe(
        audio, language="en", beam_size=5,
        word_timestamps=True, vad_filter=True,
    )
    segs = list(segments)              # generator — materialize to force the work
    dt = time.time() - t1
    _report(info.duration, dt, segs, kind="faster")


def run_mlx(model_repo, audio):
    import mlx_whisper
    print(f"[load+transcribe] mlx-whisper repo={model_repo}…", flush=True)
    t1 = time.time()
    # mlx-whisper loads the model and transcribes in one call; word_timestamps supported.
    res = mlx_whisper.transcribe(
        audio, path_or_hf_repo=model_repo,
        language="en", word_timestamps=True,
    )
    dt = time.time() - t1
    # mlx returns dict: {'text':..., 'segments':[{start,end,text,words:[{word,start,end,probability}], no_speech_prob, ...}]}
    dur = res["segments"][-1]["end"] if res.get("segments") else 0.0
    print(f"[transcribe+load] audio≈{dur:.1f}s  total={dt:.1f}s  RTF(total/audio)={dt/max(dur,1e-9):.2f}  speed={max(dur,1e-9)/dt:.1f}x")
    print("[text]", res["text"].strip())
    if res.get("segments"):
        s0 = res["segments"][0]
        print(f"[segments] {len(res['segments'])}  [no_speech_prob] {s0.get('no_speech_prob')}")
        if s0.get("words"):
            w = s0["words"][0]
            print(f"[word0] '{w['word'].strip()}' start={w['start']:.2f}s end={w['end']:.2f}s prob={w.get('probability')}")


def _report(audio_s, dt, segs, kind):
    print(f"[transcribe] audio={audio_s:.1f}s  proc={dt:.1f}s  RTF(proc/audio)={dt/audio_s:.2f}  speed={audio_s/dt:.1f}x")
    print("[text]", " ".join(s.text.strip() for s in segs))
    if segs:
        s0 = segs[0]
        print(f"[segments] {len(segs)}  [avg_logprob] {s0.avg_logprob:.3f}  [no_speech_prob] {s0.no_speech_prob:.3f}")
        if s0.words:
            w = s0.words[0]
            print(f"[word0] '{w.word.strip()}' start={w.start:.2f}s end={w.end:.2f}s prob={w.probability:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["faster", "mlx"], required=True)
    ap.add_argument("--model", required=True,
                    help="faster: e.g. large-v3-turbo | medium | small.en ; "
                         "mlx: HF repo e.g. mlx-community/whisper-large-v3-turbo")
    ap.add_argument("--audio", required=True)
    args = ap.parse_args()

    try:
        if args.backend == "faster":
            run_faster(args.model, args.audio)
        else:
            run_mlx(args.model, args.audio)
    except Exception as e:
        print(f"[ERROR] spike failed: {type(e).__name__}: {e}", file=sys.stderr)
        raise
