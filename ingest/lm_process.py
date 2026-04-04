import anthropic
import os
from music21 import chord, converter, corpus, duration, dynamics, expressions, note, stream, tempo
import dotenv
import pickle
from tqdm import tqdm
from process_features import parse_and_pickle
from process_features import parse_and_pickle

dotenv.load_dotenv()

FILE_PATH = "data/raw_xml/holst-the-planets-op-32"

PROMPT = open("ingest/process_prompt.txt").read()

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_KEY"])


def get_description(excerpt: str) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": PROMPT.format(excerpt),
            }
        ],
    )
    return response.content[0].text


_ART_SHORT = {
    "Staccato": "st",
    "Staccatissimo": "stiss",
    "Accent": "acc",
    "StrongAccent": "sacc",
    "Tenuto": "ten",
    "Marcato": "marc",
    "Fermata": "ferm",
    "Spiccato": "spic",
    "Pizzicato": "pizz",
}


def _dur_token(d: duration.Duration) -> str:
    if isinstance(d, duration.GraceDuration):
        inner = getattr(d, "duration", None)
        if inner is not None and inner is not d:
            return _dur_token(inner)
        return "g"
    t = getattr(d, "type", None)
    ql = float(d.quarterLength)
    if t in (None, "inexpressible"):
        return f"{ql:g}ql"
    abbr = {
        "whole": "w",
        "half": "h",
        "quarter": "q",
        "eighth": "e",
        "16th": "s",
        "32nd": "32",
        "64th": "64",
        "128th": "128",
        "breve": "breve",
        "longa": "longa",
    }
    s = abbr.get(t, t)
    dots = int(getattr(d, "dots", 0) or 0)
    if dots:
        s += "." * dots
    return s


def _collect_tuplets(d: duration.Duration) -> list[duration.Tuplet]:
    tups = list(getattr(d, "tuplets", None) or [])
    if not tups and isinstance(d, duration.GraceDuration):
        inner = getattr(d, "duration", None)
        if inner is not None and inner is not d:
            tups = list(getattr(inner, "tuplets", None) or [])
    return tups


def _tuplet_suffix(d: duration.Duration) -> str:
    parts: list[str] = []
    for tup in _collect_tuplets(d):
        na = getattr(tup, "numberNotesActual", None)
        nn = getattr(tup, "numberNotesNormal", None)
        if na is not None and nn is not None:
            parts.append(f"{na}:{nn}")
        elif hasattr(tup, "tupletMultiplier"):
            parts.append(str(tup.tupletMultiplier()))
    if not parts:
        return ""
    return "^" + ",".join(parts)


def _dur_with_tuplets(d: duration.Duration) -> str:
    return _dur_token(d) + _tuplet_suffix(d)


def _grace_prefix(el: note.GeneralNote) -> str:
    d = el.duration
    if isinstance(d, duration.GraceDuration) or getattr(d, "isGrace", False):
        return "g:"
    return ""


def _artic_suffix(el: note.GeneralNote) -> str:
    arts = getattr(el, "articulations", None) or []
    if not arts:
        return ""
    bits = [_ART_SHORT.get(type(a).__name__, type(a).__name__[:4].lower()) for a in arts]
    return "+" + "+".join(bits)


def _tie_suffix(el: note.GeneralNote) -> str:
    t = getattr(el, "tie", None)
    if t is not None and getattr(t, "type", None) == "start":
        return "~"
    return ""


def _event_token(el: note.GeneralNote) -> str | None:
    if getattr(el.style, "hideObjectOnPrint", False):
        return None
    gp = _grace_prefix(el)
    dur = _dur_with_tuplets(el.duration)
    tie = _tie_suffix(el)
    art = _artic_suffix(el)
    tail = f"{tie}{art}"

    if isinstance(el, note.Rest):
        return f"{gp}rest{dur}{tail}"
    if isinstance(el, chord.Chord):
        pits = "+".join(p.nameWithOctave.replace("-", "b") for p in el.pitches)
        return f"{gp}{pits}{dur}{tail}"
    if isinstance(el, note.Unpitched):
        pp = el.displayPitch()
        pstr = pp.nameWithOctave.replace("-", "b")
        return f"{gp}{pstr}{dur}{tail}"
    if isinstance(el, note.Note):
        pstr = el.pitch.nameWithOctave.replace("-", "b")
        return f"{gp}{pstr}{dur}{tail}"
    return None


def _rest_only_events(events: list[str]) -> bool:
    return all(e.removeprefix("g:").startswith("rest") for e in events)


def window_to_text(src: stream.Score, start_idx: int, size: int,
                   omit_per_part_rests: bool = True) -> str:
    lines: list[str] = []
    for part in src.parts:
        pname = (part.partName or getattr(part, "id", None) or "part").strip()
        ms = list(part.getElementsByClass(stream.Measure))
        chunk = ms[start_idx : start_idx + size]
        for m in chunk:
            flat = m.flatten()
            events: list[str] = []
            for el in sorted(
                flat.notesAndRests,
                key=lambda x: (float(x.offset or 0.0), getattr(x, "sortOrderWithinOffset", 0)),
            ):
                tok = _event_token(el)
                if tok is not None:
                    events.append(tok)
            if not events:
                line = f"{pname}: rest"
                if omit_per_part_rests:
                    continue
            elif omit_per_part_rests and _rest_only_events(events):
                continue
            else:
                line = f"{pname}: {' '.join(events)}"
            lines.append(line)
    if not lines:
        ms0 = list(src.parts[0].getElementsByClass(stream.Measure))
        ch = ms0[start_idx : start_idx + size]
        if ch:
            return "(all parts rest)"
        return "(empty window)"
    # print(lines)
    return "\n".join(lines)


def lm_process(path: str, piece: str, window_size: int = 2, start_bar: int = 1, end_bar: int | None = None, 
               omit_per_part_rests: bool = True) -> None:
    start_idx = start_bar - 1
    end_idx = None if end_bar is None else end_bar
    # end_idx = end_idx - 1 for conversion to bar index and + 1 for inclusive range
    pickle_path = path + ".pickle"
    text_path = path + ".lmdesc.txt"
    score = pickle.load(open(pickle_path, "rb"))
    if not isinstance(score, stream.Score):
        score = score.toScore()

    part0 = score.parts[0]
    measures = list(part0.getElementsByClass(stream.Measure))

    assert len(measures) >= 1
    assert window_size >= 1

    n = len(measures)
    if end_idx is None:
        end_idx = n
    if start_idx < 0 or start_idx > n:
        raise ValueError(f"start_idx must be in [0, {n}], got {start_idx}")
    if end_idx < 0 or end_idx > n:
        raise ValueError(f"end_idx must be in [0, {n}], got {end_idx}")
    if start_idx >= end_idx:
        raise ValueError(f"require start_idx < end_idx, got {start_idx} >= {end_idx}")

    for i in tqdm(range(start_idx, end_idx)):
        actual_size = min(window_size, len(measures) - i)
        start_num = measures[i].number
        excerpt = window_to_text(score, i, actual_size, omit_per_part_rests=omit_per_part_rests)
        description = get_description(excerpt)
        with open(text_path, "a") as f:
            f.write(f"{piece} [{start_num}]: {description}\n")


if __name__ == "__main__":
    lm_process(FILE_PATH, "planets", window_size=2, end_bar=185)
