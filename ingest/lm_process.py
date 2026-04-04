import anthropic
import os
from music21 import corpus, converter, stream, dynamics, expressions, tempo
import dotenv
import pickle
from tqdm import tqdm
from process_features import parse_and_pickle
from process_features import parse_and_pickle

dotenv.load_dotenv()

FILE_PATH = "data/raw_xml/holst-the-planets-op-32"

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_KEY"])


def get_description(excerpt: str) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": (
                    "Describe briefly the following bars of score, including the playing instruments, texture, feel, and any notable features\n\n"
                    + excerpt + 
                    "\n\nIn *one single* brief line with no line breaks, separate attributes of the score with semicolons (no more than 3 words each) and do not use full sentences. Do not output titles or anything other than what is asked for: \n\n"
                    "When listing instruments, only list those that are playing in the excerpt, i.e. do not state the name of the full instrumentation."
                    "Example: '3/4 time; sparse; prestissimo; snare drum; run in flutes; unison; etc.'"
                ),
            }
        ],
    )
    return response.content[0].text


def window_to_text(src: stream.Score, start_idx: int, size: int, omit_per_part_rests: bool = True) -> str:
    lines: list[str] = []
    for part in src.parts:
        pname = (part.partName or getattr(part, "id", None) or "part").strip()
        ms = list(part.getElementsByClass(stream.Measure))
        chunk = ms[start_idx : start_idx + size]
        for m in chunk:
            flat = m.flatten()
            pitch_bits: list[str] = []
            for el in flat.notes:
                if hasattr(el, "pitch"):
                    pitch_bits.append(el.pitch.nameWithOctave.replace("-", "b"))
                elif hasattr(el, "pitches"):
                    pitch_bits.append(
                        "+".join(p.nameWithOctave.replace("-", "b") for p in el.pitches)
                    )
            if not pitch_bits:
                line = f"m{m.number} {pname}: rest"
                if omit_per_part_rests:
                    continue
            else:
                line = f"m{m.number} {pname}: {' '.join(pitch_bits)}"
            lines.append(line)
    if not lines:
        ms0 = list(src.parts[0].getElementsByClass(stream.Measure))
        ch = ms0[start_idx : start_idx + size]
        if ch:
            return f"m{ch[0].number}–m{ch[-1].number}: (all parts rest)"
        return "(empty window)"
    return "\n".join(lines)


def lm_process(path: str, window_size: int = 2, max_windows: int | None = None, omit_per_part_rests: bool = True) -> None:

    pickle_path = path + ".pickle"
    text_path = path + ".lmdesc.txt"
    score = pickle.load(open(pickle_path, "rb"))
    if not isinstance(score, stream.Score):
        score = score.toScore()

    part0 = score.parts[0]
    measures = list(part0.getElementsByClass(stream.Measure))
    
    assert len(measures) >= window_size

    total = len(measures) - window_size + 1
    if max_windows is not None:
        total = min(total, max_windows)

    for i in tqdm(range(total)):
        start_num = measures[i].number
        end_num = measures[i + window_size - 1].number
        excerpt = window_to_text(score, i, window_size, omit_per_part_rests=omit_per_part_rests)
        description = get_description(excerpt)
        with open(text_path, "a") as f:
            f.write(f"measures {start_num}–{end_num}: {description}\n")


if __name__ == "__main__":
    lm_process(FILE_PATH, window_size=2, max_windows=10)
