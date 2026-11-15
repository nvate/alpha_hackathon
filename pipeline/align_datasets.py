import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Находим корень проекта (папку infinitealpha)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Теперь пути всегда будут вести в корень, откуда бы мы ни запускали скрипт
TAGS_PATH = ROOT_DIR / "data" / "materials_tags.json"
JSONL_PATH = ROOT_DIR / "data" / "alfa_material_cards_full.jsonl"
OUT_PATH = ROOT_DIR / "data" / "aligned_materials.jsonl"


def clean_material_id(raw_id: str) -> str:
    """
    Убирает префикс типа из material_id.
    'article:123-abc' -> '123-abc'
    'course:456-def' -> '456-def'
    """
    if not raw_id:
        return ""
    if ":" in raw_id:
        return raw_id.split(":", 1)[1].strip().lower()
    return raw_id.strip().lower()


def main():
    # 1. Читаем эталонный список ID из тегов (это порядок наших векторов)
    log.info(f"Читаю эталонный порядок из {TAGS_PATH}...")
    with open(TAGS_PATH, "r", encoding="utf-8-sig") as f:
        tags_data = json.load(f)

    # Создаем список ID в нужном порядке и приводим к нижнему регистру на всякий случай
    ordered_ids = [str(item.get("ItemId")).strip().lower() for item in tags_data]
    log.info(f"В файле тегов {len(ordered_ids)} записей.")

    # 2. Читаем JSONL и индексируем по очищенному ID
    log.info(f"Читаю и индексирую {JSONL_PATH}...")
    jsonl_index = {}
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                raw_id = row.get("material_id", "")
                clean_id = clean_material_id(raw_id)
                if clean_id:
                    jsonl_index[clean_id] = row
            except json.JSONDecodeError:
                log.warning(f"Ошибка JSON на строке {line_num}")

    log.info(f"В JSONL найдено уникальных ID: {len(jsonl_index)}")

    # 3. Сшиваем
    log.info("Сшиваю данные...")
    aligned_data = []
    missing_count = 0

    for uid in ordered_ids:
        if uid in jsonl_index:
            aligned_data.append(jsonl_index[uid])
        else:
            # Если статьи нет в JSONL (например, удалили или не выгрузили), создаем заглушку
            missing_count += 1
            aligned_data.append({
                "material_id": uid,
                "title": "",
                "text_clean": "",
                "text_status": "missing",
                "_is_stub": True
            })

    log.info(f"Успешно сшито: {len(aligned_data) - missing_count} статей.")
    if missing_count > 0:
        log.warning(
            f"Внимание: {missing_count} статей из тегов не найдены в JSONL. Созданы пустые заглушки (чтобы порядок не сбился).")

    # 4. Сохраняем результат
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in aligned_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    log.info(f"Готово! Сохранено в {OUT_PATH} ({len(aligned_data)} строк).")
    log.info("Теперь порядок строк в aligned_materials.jsonl 1:1 соответствует tag_vectors.npy.")


if __name__ == "__main__":
    main()