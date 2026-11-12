import json
import numpy as np
import pandas as pd
from pathlib import Path

JSONL_PATH = Path("data/aligned_materials.jsonl")
VECTORS_PATH = Path("artifacts/tag_vectors.npy")
META_PATH = Path("artifacts/materials_meta.parquet")
OUT_PATH = Path("artifacts/difficulty_scores.npy")

# Жесткие маркеры для новичков (сильно понижают)
BEGINNER_WORDS = frozenset([
    "азы", "азбука", "основы", "база", "базовый", "новичок", "начинающий",
    "для начинающих", "для чайников", "с нуля", "первый шаг", "первые шаги",
    "просто о сложном", "простыми словами", "на пальцах", "доступно",
    "шпаргалка", "памятка", "введение", "что такое", "кто такой",
    "как выбрать", "как открыть", "как начать", "как создать",
    "как зарегистрировать", "как оформить", "как получить",
    "пошаговая инструкция", "инструкция", "чек-лист"
])

# Жесткие маркеры для экспертов (сильно повышают)
EXPERT_WORDS = frozenset([
    "продвинутый", "экспертный", "сложный", "глубокий", "детальный",
    "профессиональный", "углубленный", "для профи", "тонкости", "нюансы",
    "подводные камни", "секреты", "лайфхаки", "оптимизация", "масштабирование",
    "стратегия", "анализ", "моделирование", "прогнозирование", "сценарии",
    "матрица", "метрики", "эффективность", "управление рисками",
    "реформа", "изменения в законодательстве", "поправки", "суд",
    "судебная практика", "арбитраж", "споры", "взыскание",
    "банкротство", "ликвидация", "реорганизация", "налоговый учет",
    "схема", "валютный контроль", "комплаенс", "интеграция", "автоматизация",
    "цифровизация", "трансформация", "b2b", "b2g", "тендеры", "госзакупки",
    "юнит-экономика", "финансовый анализ"
])


# ВАЖНО: убрали "115-фз" из экспертных слов. Это тема, а не сложность.

def clean_material_id(raw_id: str) -> str:
    if not raw_id: return ""
    return raw_id.split(":", 1)[1].strip().lower() if ":" in raw_id else raw_id.strip().lower()

def contains_stem(text, word):
    """Примитивный стеммер: ищет основу слова (первые 5 букв)"""
    if len(word) <= 4:
        return word in text
    # Берем первые 5 букв. 'регистрация' -> 'регис', 'регистрации' -> 'регис'
    return word[:5] in text


def text_complexity_score(row):
    title = (row.get("title") or "").lower()
    text = (row.get("text_clean") or "")
    mat_type = (row.get("material_type") or "").lower()

    score = 0.5

    # Считаем попадания с учетом морфологии
    beg_hits = sum(1 for w in BEGINNER_WORDS if contains_stem(title, w))
    exp_hits = sum(1 for w in EXPERT_WORDS if contains_stem(title, w))

    # Штрафы и бонусы
    score -= (beg_hits * 0.25)
    score += (exp_hits * 0.25)

    text_len = len(text)
    if 0 < text_len < 1500:
        score -= 0.1
    elif text_len > 10000:
        score += 0.1

    if mat_type == "course":
        score += 0.1
    elif mat_type == "webinar":
        score += 0.05

    return score


def main():
    print("Загружаю тексты и метаданные...")
    meta = pd.read_parquet(META_PATH)
    id_to_idx = {str(mid).lower(): i for i, mid in enumerate(meta["item_id"])}
    N = len(meta)

    final_scores = np.zeros(N, dtype=np.float32)
    found_count = 0

    print("Считаю сложность на основе заголовков и текста...")
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            clean_mid = clean_material_id(row.get("material_id"))

            if clean_mid in id_to_idx:
                idx = id_to_idx[clean_mid]
                final_scores[idx] = text_complexity_score(row)
                found_count += 1

    print(f"Оценено статей: {found_count} из {N}")

    # Нормализация (чтобы разброс был красивым от 0 до 1)
    # Если все оценки одинаковые, ничего не трогаем
    y_min, y_max = final_scores.min(), final_scores.max()
    if y_max > y_min:
        final_scores = 0.1 + 0.8 * (final_scores - y_min) / (y_max - y_min)

    np.save(OUT_PATH, final_scores.astype(np.float32))
    print(f"Готово! Сохранено в {OUT_PATH}")

    # Вывод для проверки
    idx_sorted = np.argsort(final_scores)
    print("\n--- Топ-5 самых простых (0.0 - новичок) ---")
    for i in idx_sorted[:5]:
        print(f"[{final_scores[i]:.2f}] {meta['title'].iloc[i][:70]}")

    print("\n--- Топ-5 самых сложных (1.0 - эксперт) ---")
    for i in idx_sorted[-5:]:
        print(f"[{final_scores[i]:.2f}] {meta['title'].iloc[i][:70]}")


if __name__ == "__main__":
    main()