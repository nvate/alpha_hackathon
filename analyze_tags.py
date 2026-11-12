"""
Аудит тегов: все уникальные значения после нормализации, с частотами.
Помогает решить, что канонизировать, а что оставить как есть.

Запуск:
    python analyze_tags.py --input materials_tags.json
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path

SEMANTIC_FIELDS = ["Topics", "Tags", "Subject", "Industries"]

LATIN_TO_CYR = str.maketrans({
    "a": "а", "e": "е", "o": "о", "p": "р", "c": "с",
    "x": "х", "y": "у", "k": "к", "m": "м", "t": "т",
    "h": "н", "b": "ь",
})


def _mostly_cyrillic(s: str, threshold: float = 0.6) -> bool:
    if not s:
        return False
    cyr = sum(1 for c in s if "а" <= c <= "я" or c == "ё")
    return cyr / len(s) >= threshold


def normalize_token(t: str) -> str:
    if not t:
        return ""
    s = unicodedata.normalize("NFC", t.strip().lower())
    s = "_".join(s.split())
    if _mostly_cyrillic(s):
        s = s.translate(LATIN_TO_CYR)
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("materials_tags.json"))
    p.add_argument("--top", type=int, default=200)
    args = p.parse_args()

    with open(args.input, encoding="utf-8-sig") as f:
        items = json.load(f)

    # Частоты по полям и общие
    by_field: dict[str, Counter] = {f: Counter() for f in SEMANTIC_FIELDS}
    overall: Counter = Counter()

    for m in items:
        for field in SEMANTIC_FIELDS:
            for raw in m.get(field) or []:
                tok = normalize_token(raw)
                if tok:
                    by_field[field][tok] += 1
                    overall[tok] += 1

    print(f"\n=== ВСЕГО УНИКАЛЬНЫХ ТОКЕНОВ: {len(overall)} ===\n")

    for field in SEMANTIC_FIELDS:
        print(f"\n--- {field} ({len(by_field[field])} уникальных) ---")
        for tok, cnt in by_field[field].most_common(args.top):
            print(f"  {cnt:4d}  {tok!r}")

    # Кандидаты на синонимию — по префиксу
    print("\n=== ПОХОЖИЕ ПРЕФИКСЫ (кандидаты в синонимы) ===")
    toks = sorted(overall)
    for i, a in enumerate(toks):
        for b in toks[i + 1:]:
            # одна строка — префикс другой, или общий префикс >= 6 символов
            common = 0
            for x, y in zip(a, b):
                if x != y:
                    break
                common += 1
            if common >= 6 and abs(len(a) - len(b)) < 15:
                print(f"  common='{a[:common]}'  {a!r} ({overall[a]})  vs  {b!r} ({overall[b]})")


if __name__ == "__main__":
    main()