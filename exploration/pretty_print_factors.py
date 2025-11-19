import argparse
import ast
import json
from pathlib import Path

import colorama
from colorama import Fore, Style


COLORS = [
    Fore.RED,
    Fore.GREEN,
    Fore.YELLOW,
    Fore.CYAN,
    Fore.MAGENTA,
    Fore.BLUE,
    Fore.LIGHTRED_EX,
    Fore.LIGHTGREEN_EX,
    Fore.LIGHTYELLOW_EX,
    Fore.LIGHTCYAN_EX,
    Fore.LIGHTMAGENTA_EX,
]


def _decode_token(token):
    if token.startswith("b'") or token.startswith('b"'):
        try:
            literal = ast.literal_eval(token)
            if isinstance(literal, bytes):
                return literal.decode("utf-8", errors="ignore")
        except Exception:
            pass
    return token


def load_records(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            tokens = record.get("tokens")
            if tokens:
                record["tokens"] = [_decode_token(tok) for tok in tokens]
            yield record


def color_for_expert(expert):
    return COLORS[expert % len(COLORS)]


def render_stage(tokens, selections):
    if tokens:
        colored = []
        for idx, token in enumerate(tokens):
            expert = selections.get(idx)
            text = token or f"[idx={idx}]"
            if expert is None:
                colored.append(text)
            else:
                colored.append(f"{color_for_expert(expert)}{text}{Style.RESET_ALL}")
        return " ".join(colored)
    colored = []
    for pos, expert in sorted(selections.items()):
        token = f"[idx={pos}]"
        colored.append(f"{color_for_expert(expert)}{token}{Style.RESET_ALL}")
    return " ".join(colored) if colored else "(no tokens)"


def main():
    parser = argparse.ArgumentParser(description="Pretty-print factor selections from JSONL.")
    parser.add_argument("--input", default="outputs/factor_analysis.jsonl", help="JSONL file produced by analyze_factors.py")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of sentences to display")
    args = parser.parse_args()

    colorama.init(autoreset=True)

    for idx, record in enumerate(load_records(args.input)):
        if args.limit is not None and idx >= args.limit:
            break

        tokens = record.get("tokens") or []
        sentence = record.get("sentence") or (" ".join(tokens) if tokens else "(no text)")
        print(f"\nSentence #{record.get('index', idx)}: {sentence}")

        for stage_info in record.get("stages", []):
            stage = stage_info["stage"]
            selections = {item["position"]: item["expert"] for item in stage_info.get("selections", [])}
            line = render_stage(tokens, selections)
            print(f"Stage {stage}: {line}")


if __name__ == "__main__":
    main()
