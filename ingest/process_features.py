import os
from music21 import corpus, converter, stream, dynamics, expressions, tempo
import pandas as pd
import pickle
from tqdm import tqdm
from collections import defaultdict

FILE_PATH = "data/raw_xml/holst-the-planets-op-32"

def parse_and_pickle(file_path: str) -> None:
    score = converter.parse(file_path)
    with open(file_path.replace(".xml", ".pickle"), "wb") as f:
        pickle.dump(score, f)

def get_express_classes(measure: stream.Measure) -> list[str]:
    classes = []
    flat = measure.flatten()

    for dyn in flat.getElementsByClass(dynamics.Dynamic):
        classes.append(dyn.value)

    for wedge in flat.getElementsByClass(dynamics.DynamicWedge):
        classes.append(wedge.type)

    for mark in flat.getElementsByClass(tempo.MetronomeMark):
        if mark.text:
            classes.append(mark.text.lower())

    for expr in flat.getElementsByClass(expressions.TextExpression):
        classes.append(expr.content.lower())

    for note in flat.notes:
        if any(isinstance(e, expressions.Fermata) for e in note.expressions):
            classes.append("fermata")
            break
    return list(dict.fromkeys(classes))


def _fmt_pitch(p) -> str:
    return p.nameWithOctave.replace('-', 'b')


def get_pitch_contour(notes: list) -> list[str]:
    pitches = [n.pitch for n in notes if hasattr(n, 'pitch')]
    if not pitches:
        return []

    midis = [p.midi for p in pitches]
    first, last = midis[0], midis[-1]
    peak, trough = max(midis), min(midis)
    peak_i, trough_i = midis.index(peak), midis.index(trough)
    span = peak - trough

    if span <= 2:
        return ["static"]
    elif peak_i > 0 and peak_i < len(midis) - 1 and peak - first > 2 and peak - last > 2:
        return ["arch"]
    elif trough_i > 0 and trough_i < len(midis) - 1 and first - trough > 2 and last - trough > 2:
        return ["valley"]
    elif last - first >= 3:
        return ["ascending"]
    elif first - last >= 3:
        return ["descending"]
    else:
        return ["meandering"]


def get_melodic_snippet(notes: list, label: str) -> str:
    """Returns a compact pitch descriptor: 'ascending G2->D3', 'static G2-D5', etc."""
    pitches = [n.pitch for n in notes if hasattr(n, 'pitch')]
    if not pitches:
        return ""
    low  = _fmt_pitch(min(pitches, key=lambda p: p.midi))
    high = _fmt_pitch(max(pitches, key=lambda p: p.midi))
    if label == "static":
        return f"static {low}" if low == high else f"static {low}-{high}"
    first = _fmt_pitch(pitches[0])
    last  = _fmt_pitch(pitches[-1])
    return f"{label} {first}->{last}"


def get_pitch_range(notes: list) -> str:
    pitches = [n.pitch for n in notes if hasattr(n, 'pitch')]
    if not pitches:
        return ""
    low  = _fmt_pitch(min(pitches, key=lambda p: p.midi))
    high = _fmt_pitch(max(pitches, key=lambda p: p.midi))
    return low if low == high else f"{low}–{high}"


# sustained, runs, triplet, irregular, etc.
def get_rhythm_classes(measure: stream.Measure) -> list[str]:
    all_notes = list(measure.flatten().notes)
    attacks = [n for n in all_notes if not (n.tie and n.tie.type == 'stop')]

    if not attacks:
        return ["held"]  # bar is entirely tied continuation from previous bar

    tempo_mark = measure.getElementsByClass('MetronomeMark').first()
    bps = tempo_mark.number / 60 if tempo_mark else 2.0

    durations = [n.duration.quarterLength / bps for n in attacks]
    avg_dur = sum(durations) / len(durations)
    min_dur = min(durations)
    max_dur = max(durations)

    classes = []
    if any(n.duration.tuplets and n.duration.tuplets[0].numberNotesActual not in (2, 3) for n in attacks):
        classes.append("irregular")
    if any(n.duration.tuplets and n.duration.tuplets[0].numberNotesActual == 3 for n in attacks):
        classes.append("triplet")
    if max_dur >= 1.0:
        classes.append("slow")
    if avg_dur < 0.75:
        classes.append("fast")
    if max_dur >= 1.0 and min_dur < 0.5:
        classes.append("mixed")

    return classes or ["mixed"]



def medium_convert(score: stream.Score, head_length: int = None) -> pd.DataFrame:
    measure_descriptions = []
    n_measures = len(score.parts[0].getElementsByClass(stream.Measure))
    print(f"Found {n_measures} measures")
    families = defaultdict(list)
    STRING_NAMES = {"Violins I", "Violins II", "Violas", "Violoncellos", "Contrabasses"}
    for part in score.parts:
        inst = part.getInstrument()
        if inst.instrumentSound:
            key = inst.instrumentSound
        elif part.partName in STRING_NAMES:
            key = part.partName
        else:
            continue
        families[key].append(part)

    

    for i in tqdm(range(head_length if head_length and head_length <= n_measures else n_measures)):
        row = {"measure": i + 1}
        for family, parts in families.items():
            rhythm_classes = set()
            express_classes = set()
            contour_snippets = []
            nonempty_count = 0
            total_note_nums = 0
            total_avg_dur = 0.0
            for part in parts:
                measure = part.getElementsByClass(stream.Measure)[i]
                all_notes = list(measure.flatten().notes)
                attacks = [n for n in all_notes if not (n.tie and n.tie.type == 'stop')]
                express_classes.update(get_express_classes(measure))
                if not all_notes:
                    continue
                tempo_mark = measure.getElementsByClass('MetronomeMark').first()
                bps = tempo_mark.number / 60 if tempo_mark else 2.0
                nonempty_count += 1
                if attacks:
                    avg_dur = sum(n.duration.quarterLength / bps for n in attacks) / len(attacks)
                    total_note_nums += len(attacks)
                    total_avg_dur += avg_dur
                rhythm_classes.update(get_rhythm_classes(measure))
                label = get_pitch_contour(attacks)[0] if attacks and get_pitch_contour(attacks) else ""
                snippet = get_melodic_snippet(attacks, label) if label else ""
                if snippet and snippet not in contour_snippets:
                    contour_snippets.append(snippet)

            if nonempty_count > 0:
                avg_notes = total_note_nums / nonempty_count
                avg_dur = total_avg_dur / nonempty_count
                player_str = f"{nonempty_count}/{len(parts)} players" if len(parts) > 1 else ""
                parts_str = f"{player_str + ' | ' if player_str else ''}avg {avg_notes:.1f} notes @ {avg_dur:.2f}s | {','.join(rhythm_classes)}"
                if contour_snippets:
                    parts_str += f" | {'; '.join(contour_snippets)}"
                if express_classes:
                    parts_str += f" | {','.join(express_classes)}"
                row[family] = parts_str
            else:
                row[family] = "rest"
        measure_descriptions.append(row)
    
    return pd.DataFrame(measure_descriptions)



# parse_and_pickle(FILE_PATH + ".xml")


def save_instrument_docs(df: pd.DataFrame, file_path: str) -> None:
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    out_dir = os.path.join(os.path.dirname(file_path), base_name)
    os.makedirs(out_dir, exist_ok=True)

    instrument_cols = [c for c in df.columns if c != "measure"]
    for instrument in instrument_cols:
        lines = [f"Instrument: {instrument}\n"]
        for _, row in df.iterrows():
            lines.append(f"Measure {int(row['measure'])}: {row[instrument]}")
        out_path = os.path.join(out_dir, f"{instrument}.txt")
        with open(out_path, "w") as f:
            f.write("\n".join(lines))

    print(f"Saved {len(instrument_cols)} instrument docs to {out_dir}/")


# parse_and_pickle(FILE_PATH + ".xml")


# score = pickle.load(open(FILE_PATH + ".pickle", "rb"))
# print(f"Loaded {score}")
# df = medium_convert(score, head_length=150)
# df.to_csv(FILE_PATH + ".csv", index=False)
# save_instrument_docs(df, FILE_PATH + ".xml")


df = pd.read_csv(FILE_PATH + ".csv")
save_instrument_docs(df, FILE_PATH)