import json
from pathlib import Path

LEVEL_MAP_PATH = Path("data/level_mapping.json")


def parse_form_data(form_data: dict) -> dict:
    """
    Принимает сырые данные из request.form.
    Возвращает чистый профиль для алгоритма графа (БЕЗ ограничений по цене).
    """
    user_level_str = form_data.get("level", "zero")

    if user_level_str == "zero":
        user_difficulty = 0.1
    elif user_level_str == "basic":
        user_difficulty = 0.45
    elif user_level_str == "some":
        user_difficulty = 0.55
    else:
        user_difficulty = 0.60

    direction = form_data.get("direction", "tax_law")

    direction_to_tag = {
        "start_strategy": "старт_бизнеса",
        "finance_investments": "финансы_и_учет",
        "tax_law": "налоги_и_законы",
        "marketing_brand": "маркетинг",
        "digital_marketing": "маркетинг",
        "sales_clients": "продажи_и_переговоры",
        "analitics_data": "анализ_данных",
        "team_hr": "управление_командой",
        "product_management": "управление_командой",
        "product_dev": "технологии",
        "tools": "технологии"
    }

    target_tag = direction_to_tag.get(direction, "старт_бизнеса")

    return {
        "user_difficulty": user_difficulty,
        "target_tag": target_tag,
        "raw_goal": form_data.get("description", "")
    }