import csv
import json
import sys

import numpy as np

LATIN = str.maketrans("aceopxyk", "асеорхук")


def norm(s):
    return " ".join(str(s).lower().translate(LATIN).replace("ё", "е").split())


def main(tags_path, scores_path, out_path, mapping_path="data/level_mapping.json",
         directions_path="data/direction_topics.json"):
    items = json.load(open(tags_path, encoding="utf-8"))
    scores = np.load(scores_path)
    if len(items) != len(scores):
        sys.exit(f"Число материалов ({len(items)}) не совпадает с числом оценок ({len(scores)})")
    levels = json.load(open(mapping_path, encoding="utf-8"))["levels"]
    dirs = json.load(open(directions_path, encoding="utf-8"))

    def level_of(x):
        for code, lv in levels.items():
            if x <= lv["max_difficulty"] + 1e-6:
                return code
        return list(levels)[-1]

    item_level = [level_of(x) for x in scores]
    labels = [{norm(t) for f in ("Tags", "Topics") for t in (it.get(f) or [])} for it in items]

    rows = []
    for d, names in [("весь каталог", None)] + list(dirs.items()):
        if names is None:
            idx = range(len(items))
        else:
            want = {norm(n) for n in names}
            idx = [i for i, l in enumerate(labels) if l & want]
        row = {"направление": d, "материалов": len(idx)}
        for code in levels:
            row[code] = sum(1 for i in idx if item_level[i] == code)
        rows.append(row)

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{'направление':22} {'всего':>6}", *[f"{c:>8}" for c in levels])
    for r in rows:
        print(f"{r['направление']:22} {r['материалов']:6}", *[f"{r[c]:8}" for c in levels])


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("Запуск: python level_coverage.py materials_tags.json artifacts/difficulty_scores.npy data/level_coverage.csv")
    main(*sys.argv[1:])
