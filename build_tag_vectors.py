"""
Векторизация материалов по структурированным полям через TF-IDF
С КАНОНИЗАЦИЕЙ СИНОНИМОВ.

Вход:  materials_tags.json
Выход: artifacts/tag_vectors.npy       (N, V) float32, L2-норма
       artifacts/material_ids.json
       artifacts/tag_vocab.json
       artifacts/synonym_map.json      — применённый маппинг (для воспроизводимости)
       artifacts/materials_meta.parquet

Запуск:
    python build_tag_vectors.py --input materials_tags.json
    python build_tag_vectors.py --input materials_tags.json --no-synonyms  # без канонизации
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
log = logging.getLogger(__name__)

SEMANTIC_FIELDS = ["Topics", "Tags", "Subject", "Industries"]
DEFAULT_INPUT   = Path("materials_tags.json")
DEFAULT_OUT_DIR = Path("artifacts")

LATIN_TO_CYR = str.maketrans({
    "a": "а", "e": "е", "o": "о", "p": "р", "c": "с",
    "x": "х", "y": "у", "k": "к", "m": "м", "t": "т",
    "h": "н", "b": "ь",
})


# ============================================================
# КАРТА СИНОНИМОВ
# Ключ — токен ПОСЛЕ normalize_token (lowercase, underscore, гомоглифы выправлены).
# Значение — канонический токен или None (дропнуть).
# Уровни уверенности помечены комментариями.
# ============================================================
SYNONYM_MAP: dict[str, str | None] = {
    # --- ✅ УВЕРЕННО: разные написания одной сущности ---

    # Налоги и законы (Topic "Законы" 35, Subject "Налоги" 78,
    #                   Subject "Изменения в законах" 1) -> один канон
    "законы":               "налоги_и_законы",
    "налоги":               "налоги_и_законы",
    "изменения_в_законах":  "налоги_и_законы",

    # Финансы (Topic "Финансы" 11, Subject "Финансы" 38)
    "финансы":              "финансы_и_учет",

    # Старт бизнеса (Topic "Первые шаги" 4)
    "первые_шаги":          "старт_бизнеса",

    # Аналитика (Tag "Работа с данными" 1)
    "работа_с_данными":     "анализ_данных",

    # HoReCa — гомоглиф, автозамена не ловит (4 латинских + 1 кириллическая 'с')
    "horeсa":               "horeca",

    # Шум: маркеры "новинка"
    "new":                  None,
    "💥_new":               None,

    # --- ⚠️ СПОРНО: включено, но помечено ---
    "продажи":              "продажи_и_переговоры",  # спорно: узкий навык vs широкая тема

    # --- ⚠️ ОТКЛЮЧЕНО: риск перекоса ---
    # "законы_и_государство": "налоги_и_законы",  # раскомментируй после сравнения
}

# Стоп-слова для title: — служебные слова и «мусорные» частоты
TITLE_STOPWORDS = {
    # служебные
    "как", "что", "для", "или", "при", "без", "где", "это", "этот", "эта",
    "если", "когда", "чем", "кто", "так", "уже", "еще", "ещё", "они", "она",
    "оно", "его", "её", "их", "все", "весь", "всё", "вам", "вас", "мы", "вы",
    "наш", "ваш", "свой", "который", "которая", "которые", "чтобы", "только",
    "тоже", "также", "быть", "был", "была", "было", "были", "будет", "есть",
    # предлоги/союзы
    "на", "по", "из", "от", "до", "со", "ко", "не", "ни", "но", "да", "нет",
    "за", "под", "над", "про", "об", "обо", "в", "во", "с", "к", "у", "о",
    "эксперт", "экспертом", "экспертами",
    "выпуск",
    "работать", "работает",
    "календарь", "производственный",
    "разбираемся", "разбираем",
    "год", "году", "года", "годах", "2024", "2025", "2026", "2027",
}
TITLE_MIN_LEN = 4          # короче — пропускаем
TITLE_WEIGHT  = 1.0        # вес title-токенов в документе (можно 0.5 для ослабления)

# ============================================================
# НОРМАЛИЗАЦИЯ И КАНОНИЗАЦИЯ
# ============================================================
def _mostly_cyrillic(s: str, threshold: float = 0.6) -> bool:
    if not s:
        return False
    cyr = sum(1 for c in s if "а" <= c <= "я" or c == "ё")
    return cyr / len(s) >= threshold


def normalize_token(t: str) -> str:
    """
    Приводит значение поля к «сырой» канонической форме:
    lowercase, пробелы -> underscore, гомоглифы -> кириллица.
    """
    if not t:
        return ""
    s = unicodedata.normalize("NFC", t.strip().lower())
    s = "_".join(s.split())
    if _mostly_cyrillic(s):
        s = s.translate(LATIN_TO_CYR)
    return s


def canonicalize(token: str, use_synonyms: bool = True) -> str | None:
    """
    Применяет SYNONYM_MAP. Возвращает None, если токен надо дропнуть.
    При use_synonyms=False — no-op (для отладки и для 'что было бы без').
    """
    if not use_synonyms:
        return token
    return SYNONYM_MAP.get(token, token)


def _title_tokens(title: str) -> list[str]:
    """Разбивает заголовок на нормализованные значимые слова с префиксом title:."""
    if not title:
        return []
    out = []
    for word in title.lower().split():
        # оставляем только буквы/цифры, дефисы и подчёркивания
        w = "".join(c for c in word if c.isalnum() or c in "-_")
        if len(w) < TITLE_MIN_LEN:
            continue
        if w in TITLE_STOPWORDS:
            continue
        # гомоглифы — та же логика, что для тегов
        if _mostly_cyrillic(w):
            w = w.translate(LATIN_TO_CYR)
        out.append(f"title:{w}")
    return out

def build_doc(m: dict, use_synonyms: bool = True,
              use_title: bool = True) -> str:
    """
    Документ = множество канонических тегов + (опционально) токены из Title.
    title-токены помечены префиксом, чтобы TF-IDF учитывал их отдельно от тегов.
    """
    tokens: set[str] = set()
    for field in SEMANTIC_FIELDS:
        for t in m.get(field) or []:
            norm = normalize_token(t)
            if not norm:
                continue
            canon = canonicalize(norm, use_synonyms)
            if canon is None:
                continue
            tokens.add(canon)

    if use_title:
        for tt in _title_tokens(m.get("Title") or ""):
            tokens.add(tt)

    # TITLE_WEIGHT > 1.0 — дублируем title-токены в строке, чтобы поднять их вес
    # (sublinear_tf сглаживает, так что даже простое добавление в set уже даёт эффект)
    out = []
    for tok in sorted(tokens):
        if tok.startswith("title:") and TITLE_WEIGHT > 1.0:
            out.extend([tok] * int(TITLE_WEIGHT))
        else:
            out.append(tok)
    return " ".join(out)


# ============================================================
# MAIN
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",       type=Path, default=DEFAULT_INPUT)
    p.add_argument("--out-dir",     type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--min-df",      type=int,  default=1)
    p.add_argument("--no-title", action="store_true",
                   help="Не использовать токены из Title")
    p.add_argument("--no-synonyms", action="store_true",
                   help="Отключить канонизацию (для сравнения)")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    use_syn = not args.no_synonyms
    use_title = not args.no_title

    # --- 1. Загрузка ---
    with open(args.input, encoding="utf-8-sig") as f:
        items = json.load(f)
    log.info("Материалов: %d", len(items))
    log.info("Канонизация синонимов: %s", "ON" if use_syn else "OFF")
    log.info("Title-токены: %s", "ON" if use_title else "OFF")

    # --- 2. Сборка документов ---
    docs, ids, titles, types = [], [], [], []
    n_raw, n_canon, is_empty = [], [], []
    canon_stats: Counter = Counter()  # какие маппинги реально сработали


    for m in items:
        docs.append(build_doc(m, use_syn, use_title))
        ids.append(m.get("ItemId") or "")
        titles.append(m.get("Title") or "")
        types.append(m.get("ItemType") or "")

        # диагностика по кол-ву токенов до/после
        raw_tokens = set()
        for field in SEMANTIC_FIELDS:
            for t in m.get(field) or []:
                norm = normalize_token(t)
                if norm:
                    raw_tokens.add(norm)

        if use_syn:
            canon_tokens = set()
            for t in raw_tokens:
                c = canonicalize(t, use_synonyms=True)
                if c is None:
                    canon_stats[f"DROPPED:{t}"] += 1
                    continue
                if c != t:
                    canon_stats[f"{t} -> {c}"] += 1
                canon_tokens.add(c)
            n_raw.append(len(raw_tokens))
            n_canon.append(len(canon_tokens))
            is_empty.append(len(canon_tokens) == 0)
        else:
            n_raw.append(len(raw_tokens))
            n_canon.append(len(raw_tokens))
            is_empty.append(len(raw_tokens) == 0)

    if canon_stats:
        log.info("Сработавшие канонизации:")
        for key, cnt in canon_stats.most_common():
            log.info("  %-40s %d", key, cnt)

    n_empty = sum(is_empty)
    if n_empty:
        log.warning("Материалов с пустым набором токенов: %d "
                    "(получат нулевые векторы)", n_empty)

    if use_syn:
        shrunk = sum(1 for a, b in zip(n_raw, n_canon) if b < a)
        log.info("Материалов, где кол-во токенов уменьшилось: %d из %d",
                 shrunk, len(items))

    # --- 3. TF-IDF ---
    log.info("Считаю TF-IDF...")
    vectorizer = TfidfVectorizer(
        token_pattern=r"\S+",
        lowercase=False,
        min_df=args.min_df,
        sublinear_tf=True,
        norm="l2",
    )
    X_sparse = vectorizer.fit_transform(docs)
    vocab = vectorizer.vocabulary_
    log.info("Векторы: shape=%s, словарь=%d токенов", X_sparse.shape, len(vocab))

    X = X_sparse.toarray().astype(np.float32)
    norms = np.linalg.norm(X, axis=1)
    log.info("Нормы: min=%.4f, mean=%.4f, max=%.4f, нулевых=%d",
             norms.min(), norms.mean(), norms.max(), int((norms == 0).sum()))

    # Сколько векторов идентичны друг другу — прямой индикатор «клик»
    # (после L2-нормы: одинаковые наборы токенов дают одинаковые векторы)
    from collections import defaultdict
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, doc in enumerate(docs):
        if doc:
            groups[tuple(sorted(doc.split()))].append(i)
    dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
    n_in_dups = sum(len(v) for v in dup_groups.values())
    log.info("Групп идентичных векторов: %d, суммарно материалов в них: %d",
             len(dup_groups), n_in_dups)
    if dup_groups:
        biggest = sorted(dup_groups.items(), key=lambda kv: -len(kv[1]))[:3]
        for toks, members in biggest:
            log.info("  группа %d материалов, токены: %s",
                     len(members), list(toks))

    # --- 4. Топы IDF ---
    idf = vectorizer.idf_
    top_rare = sorted(vocab.items(), key=lambda kv: -idf[kv[1]])[:10]
    top_freq = sorted(vocab.items(), key=lambda kv: idf[kv[1]])[:10]
    log.info("Топ различающих (высокий IDF): %s", [t for t, _ in top_rare])
    log.info("Топ частых (низкий IDF):     %s", [t for t, _ in top_freq])

    # --- 5. Сохранение ---
    np.save(args.out_dir / "tag_vectors.npy", X)

    with open(args.out_dir / "material_ids.json", "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False, indent=2)

    with open(args.out_dir / "tag_vocab.json", "w", encoding="utf-8") as f:
        json.dump({t: int(i) for t, i in vocab.items()},
                  f, ensure_ascii=False, indent=2, sort_keys=True)

    # Сохраняем маппинг, чтобы графовая команда знала, что было схлопнуто
    with open(args.out_dir / "synonym_map.json", "w", encoding="utf-8") as f:
        json.dump({"applied": use_syn, "map": SYNONYM_MAP},
                  f, ensure_ascii=False, indent=2)

    pd.DataFrame({
        "row_index":         np.arange(len(ids)),
        "item_id":           ids,
        "title":             titles,
        "item_type":         types,
        "n_raw_tags":        n_raw,
        "n_canonical_tags":  n_canon,
        "is_empty":          is_empty,
    }).to_parquet(args.out_dir / "materials_meta.parquet", index=False)

    log.info("Сохранено в %s:", args.out_dir)
    log.info("  tag_vectors.npy       %s", X.shape)
    log.info("  material_ids.json     %d id", len(ids))
    log.info("  tag_vocab.json        %d токенов", len(vocab))
    log.info("  synonym_map.json      %d правил", len(SYNONYM_MAP))
    log.info("  materials_meta.parquet")


if __name__ == "__main__":
    main()