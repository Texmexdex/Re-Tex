"""Music Flamingo listening worker - runs in ITS OWN Python (config "listen" -> "python": a venv with torch +
transformers >= 5.19 + xgrammar), never the app's: it imports nothing from mashlab. listen.py starts it as a script:

    <listen python> listen_worker.py job.json out.jsonl

job.json: {"model": "<hf id>", "items": [{"id": ..., "clip": "<16 kHz mono wav>", "about": "<title, folder, BPM, key>"}]}
out.jsonl: one line per item as it finishes - {"id", "genre", "subgenre", "sound"} or {"id", "error"} - then
{"done": true}. Settings as tested in the Oct 15 blind test (Music Flamingo won 8 of 14 against the text model):
schema-constrained JSON with no free whitespace (else it pads with tabs), the plain-PyTorch grammar mask (Triton
isn't on Windows), repetition penalty 1.1 (else it loops).
"""
import json
import sys
import time

GENRES = ["Alternative", "Ambient", "Blues", "Classical", "Country", "Dance", "Drum & Bass", "Dubstep", "Electronic",
          "Experimental", "Folk", "Funk", "Hardcore", "Hip-Hop", "House", "Indie", "Jazz", "Latin", "Metal", "Pop",
          "Post-Rock", "Progressive", "Punk", "R&B", "Reggae", "Rock", "Soul", "Soundtrack", "Techno", "Trap"]
SYSTEM = ("You describe songs for a DJ's library: the main genre (one of the list), a more specific sub-genre, and "
          "one sentence (at most 30 words) on how it sounds - instruments, vocals, mood, energy. Describe what you "
          "hear; don't guess the singer's language or nationality.")
SCHEMA = {"type": "object", "properties": {"genre": {"type": "string", "enum": GENRES}, "subgenre": {"type": "string"},
                                           "sound": {"type": "string"}}, "required": ["genre", "subgenre", "sound"]}


def main(job_path, out_path):
    import functools
    import torch
    import xgrammar as xgr
    from transformers import AutoProcessor, MusicFlamingoForConditionalGeneration

    job = json.loads(open(job_path, encoding="utf-8").read())
    out = open(out_path, "a", encoding="utf-8")

    def emit(d):
        out.write(json.dumps(d, ensure_ascii=False) + "\n")
        out.flush()

    t = time.time()
    proc = AutoProcessor.from_pretrained(job["model"])
    model = MusicFlamingoForConditionalGeneration.from_pretrained(job["model"], dtype=torch.bfloat16, device_map="cuda").eval()
    xgr.apply_token_bitmask_inplace = functools.partial(xgr.apply_token_bitmask_inplace, backend="torch_native")
    info = xgr.TokenizerInfo.from_huggingface(proc.tokenizer, vocab_size=model.config.text_config.vocab_size)
    grammar = xgr.GrammarCompiler(info).compile_json_schema(json.dumps(SCHEMA), any_whitespace=False)
    emit({"loaded": round(time.time() - t, 1)})
    def ask(it, penalty):
        conv = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [{"type": "text", "text": "The song (30 s from its middle):"},
                                             {"type": "audio", "path": it["clip"]},
                                             {"type": "text", "text": "Use what you hear as well as the notes below.\n\n"
                                                                      "Describe this song. " + it.get("about", "")}]}]
        inputs = proc.apply_chat_template(conv, tokenize=True, add_generation_prompt=True, return_dict=True).to(model.device)
        inputs["input_features"] = inputs["input_features"].to(model.dtype)
        n_in = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=300, do_sample=False, repetition_penalty=penalty,
                                 logits_processor=[xgr.contrib.hf.LogitsProcessor(grammar)])
        return proc.tokenizer.decode(ids[0, n_in:], skip_special_tokens=True)

    for it in job["items"]:
        try:
            text = ask(it, 1.1)
            try:
                r = json.loads(text)
            except ValueError:                             # it ran on listing adjectives past the cap: once more,
                text = ask(it, 1.3)                        # firmer on repeats, else keep what it said up to the cut
                try:
                    r = json.loads(text)
                except ValueError:
                    r = salvage(text)
            emit({"id": it["id"], "genre": r["genre"], "subgenre": r["subgenre"].strip()[:80],
                  "sound": " ".join(r["sound"].split())[:400]})
        except Exception as e:                             # one song failing doesn't stop the rest
            emit({"id": it["id"], "error": f"{type(e).__name__}: {e}"[:300]})
    emit({"done": True})


def salvage(text, words=30):
    """A reply cut off inside its sound line: genre and sub-genre as given, the sound up to its last complete phrase
    (at most `words` words, repeated words dropped)."""
    import re
    m = re.match(r'\s*\{"genre":"([^"]*)","subgenre":"([^"]*)","sound":"(.*)', text, re.S)
    if not m:
        raise ValueError("no usable answer")
    seen, kept = set(), []
    for w in m.group(3).split()[:words * 2]:
        k = w.strip(",.;").lower()
        if k and k in seen:
            continue
        seen.add(k)
        kept.append(w)
    sound = " ".join(kept[:words])
    cut = max(sound.rfind(". "), sound.rfind("; "), sound.rfind(", "))
    sound = (sound[:cut] if cut > 20 else sound).rstrip(",;. ") + "."
    return {"genre": m.group(1), "subgenre": m.group(2), "sound": sound}


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
