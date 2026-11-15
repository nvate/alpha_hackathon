import csv
import difflib
import json
import sys
from collections import Counter, defaultdict

# Пункты шага 2 и технические навыки со скриншотов формы.
# Список со скриншотов неполный: допишите пункты, которые не попали в кадр.
KNOWN_TOPICS = {
    "Бизнес и стратегия": ["Продажи", "Переговоры", "Работа с клиентами", "Постановка целей"],
    "Маркетинг и продвижение": ["Маркетинг", "Продвижение в соцсетях", "Анализ ЦА", "Анализ рынка"],
    "Финансы и аналитика": ["Управление финансами", "Финансовый анализ", "Анализ данных", "SQL", "Excel / Google Sheets"],
    "Управление и процессы": ["Управление командой", "Управление проектами", "Постановка задач", "Работа в CRM"],
    "Технические навыки": ["Python", "JavaScript", "HTML/CSS", "Git", "Английский"],
}

# Ручные соответствия там, где название в форме отличается от названия тега в каталоге.
# Правьте этот словарь по результатам: значение может быть списком тегов каталога.
MANUAL = {
    "Переговоры": ["Ведение переговоров"],
    "Работа с клиентами": ["Коммуникация с клиентами"],
    "Управление финансами": ["Финансы и учет"],
    "Продвижение в соцсетях": ["Маркетинг"],
}


LATIN = str.maketrans("aceopxyk", "асеорхук")  # латинские двойники букв в данных


def norm(s):
    return " ".join(str(s).lower().translate(LATIN).replace("ё", "е").split())


def main(tags_path, directions_path, out_path):
    items = json.load(open(tags_path, encoding="utf-8"))
    # индексы каталога: название -> множество материалов
    catalog = defaultdict(set)
    original = {}
    for it in items:
        for field in ("Tags", "Topics"):
            for t in it.get(field) or []:
                catalog[norm(t)].add(it["ItemId"])
                original[norm(t)] = t
    names = list(catalog)

    rows = []

    def add(level, group, form_item, form_tags=None):
        wanted = MANUAL.get(form_item) or [form_item]
        found, kind = [], "нет совпадения"
        for w in wanted:
            k = norm(w)
            if k in catalog:
                found.append(original[k])
                kind = "ручное" if form_item in MANUAL else "точное"
        if not found:
            close = difflib.get_close_matches(norm(form_item), names, n=2, cutoff=0.6)
            if close:
                found = [original[c] for c in close]
                kind = "похожее, проверить"
        ids = set()
        for f in found:
            ids |= catalog[norm(f)]
        rows.append({
            "уровень": level, "группа": group, "пункт формы": form_item,
            "тег каталога": "; ".join(found), "тип совпадения": kind,
            "материалов": len(ids),
        })

    # направления из CSV формы: №, Категория, Теги внутри
    with open(directions_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            direction = r["Категория"]
            tags = [t.strip() for t in r["Теги внутри"].split(",") if t.strip()]
            start = len(rows)
            for t in tags:
                add("тег направления", direction, t)
            # само направление не тег каталога: считаем его по тегам внутри
            ids = set()
            for r2 in rows[start:]:
                for f2 in r2["тег каталога"].split("; "):
                    if f2:
                        ids |= catalog[norm(f2)]
            rows.insert(start, {"уровень": "направление", "группа": direction, "пункт формы": direction,
                                "тег каталога": "(по тегам внутри)", "тип совпадения": "сумма", "материалов": len(ids)})

    for group, topics in KNOWN_TOPICS.items():
        for t in topics:
            add("отмеченная тема", group, t)

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    cnt = Counter(r["тип совпадения"] for r in rows)
    print("Пунктов формы:", len(rows))
    for k, v in cnt.items():
        print(f"  {k}: {v}")
    empty = [r["пункт формы"] for r in rows if r["материалов"] == 0]
    print("Без материалов:", ", ".join(empty) if empty else "нет")
    used = {norm(t) for r in rows for t in r["тег каталога"].split("; ") if t and t != "(по тегам внутри)"}
    left = sorted(original[n] for n in names if n not in used)
    print("Теги каталога, не попавшие ни в один пункт формы:", len(left))
    for t in left:
        print("  ", t)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("Запуск: python build_form_mapping.py materials_tags.json form_directions.csv form_mapping.csv")
    main(*sys.argv[1:])
