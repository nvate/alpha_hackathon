import json
import numpy as np
import pandas as pd
from pathlib import Path
import networkx as nx

# --- Пути к артефактам ---
ROOT_DIR = Path(__file__).resolve().parent.parent

VECTORS_PATH = ROOT_DIR / "artifacts" / "tag_vectors.npy"
DIFFICULTY_PATH = ROOT_DIR / "artifacts" / "difficulty_scores.npy"
META_PATH = ROOT_DIR / "artifacts" / "materials_meta.parquet"

CARDS_PATH = ROOT_DIR / "data" / "aligned_materials.jsonl"

OUT_EDGES_PATH = ROOT_DIR / "artifacts" / "graph_edges.parquet"
OUT_NODES_PATH = ROOT_DIR / "artifacts" / "graph_nodes.parquet"

# --- Параметры графа ---
TOP_K = 20
MIN_SIMILARITY = 0.15
DEFAULT_DURATION_MIN = 10.0


def clean_material_id(raw_id):
    """'article:123-abc' -> '123-abc'"""
    if not raw_id:
        return ""
    return raw_id.split(":", 1)[1].strip().lower() if ":" in raw_id else raw_id.strip().lower()


def card_price(card):
    """
    Цена материала для пользователя в рублях и откуда она взята.
      purchase     -> цена разовой покупки
      subscription -> основной тариф подписки за месяц (полная цена, без скидки)
      иначе        -> 0.0, материал бесплатный
    """
    pricing = card.get("pricing") or {}
    kind = pricing.get("kind")

    if kind == "purchase":
        price = pricing.get("purchase_price_raw")
        if price is None:
            tariffs = pricing.get("purchase_tariffs") or []
            price = tariffs[0].get("price_raw") if tariffs else None
        if price is not None:
            return float(price), "purchase"

    if kind == "subscription":
        tariffs = pricing.get("subscription_tariffs") or []
        main = [t for t in tariffs if t.get("is_main")] or tariffs
        if main and main[0].get("price_raw") is not None:
            return float(main[0]["price_raw"]), "subscription_month"

    return 0.0, "free"


def card_duration(card):
    """Длительность в минутах или None, если в карточке её нет."""
    value = card.get("duration_minutes")
    if value is None or value != value or value <= 0:  # None, NaN или ноль
        return None
    return float(value)


def load_cards(meta):
    """Читает карточки и возвращает их списком в порядке строк meta."""
    by_id = {}
    with open(CARDS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                by_id[clean_material_id(row.get("material_id"))] = row
    cards = []
    for item_id in meta["item_id"]:
        card = by_id.get(str(item_id).strip().lower())
        if card is None:
            raise SystemExit(f"Для материала {item_id} нет карточки в {CARDS_PATH}. "
                             f"Запустите align_datasets.py.")
        cards.append(card)
    return cards


def price_and_duration(meta):
    """Цена, её источник, длительность и признак «длительность оценена» для каждой вершины."""
    cards = load_cards(meta)

    prices, price_kinds = zip(*(card_price(c) for c in cards))
    durations = [card_duration(c) for c in cards]
    types = list(meta["item_type"])

    # медиана известной длительности по типу материала
    known = {}
    for t, d in zip(types, durations):
        if d is not None:
            known.setdefault(t, []).append(d)
    median = {t: float(np.median(v)) for t, v in known.items()}

    estimated = [d is None for d in durations]
    filled = [d if d is not None else median.get(t, DEFAULT_DURATION_MIN)
              for d, t in zip(durations, types)]
    return list(prices), list(price_kinds), filled, estimated


def main():
    print("Загружаю данные...")
    X = np.load(VECTORS_PATH)
    difficulty = np.load(DIFFICULTY_PATH)
    meta = pd.read_parquet(META_PATH)

    N = len(meta)
    print(f"Вершин в графе: {N}")

    print("Считаю матрицу сходства...")
    sim = X @ X.T

    print("Строю направленные рёбра (строгий DAG)...")
    edges = []

    for i in range(N):
        neighbors = np.argsort(-sim[i])[1:TOP_K + 1]

        for j in neighbors:
            weight = sim[i, j]
            if weight < MIN_SIMILARITY:
                continue

            diff_i = difficulty[i]
            diff_j = difficulty[j]

            # СТРОГИЙ DAG: никаких двусторонних связей!
            if diff_j > diff_i:
                edges.append({"source": i, "target": j, "weight": float(weight)})
            elif diff_i > diff_j:
                edges.append({"source": j, "target": i, "weight": float(weight)})
            else:
                # Тай-брейк: если сложность равна, используем индекс для направления
                if i < j:
                    edges.append({"source": i, "target": j, "weight": float(weight)})
                else:
                    edges.append({"source": j, "target": i, "weight": float(weight)})

    df_edges = pd.DataFrame(edges)
    df_edges = df_edges.drop_duplicates(subset=["source", "target"], keep="first")
    n_before = len(df_edges)
    df_edges = df_edges[df_edges["source"] != df_edges["target"]].reset_index(drop=True)
    n_self_loops = n_before - len(df_edges)
    if n_self_loops > 0:
        print(f"Удалено петель (source==target): {n_self_loops}", flush=True)
    print(f"Сгенерировано рёбер: {len(df_edges)}")

    # 3. Таблица вершин
    df_nodes = meta[["row_index", "item_id", "title", "item_type"]].copy()
    df_nodes["difficulty"] = difficulty
    price, price_kind, duration, duration_estimated = price_and_duration(meta)
    df_nodes["price"] = price                        # рубли; 0 = бесплатно
    df_nodes["price_kind"] = price_kind              # free / purchase / subscription_month
    df_nodes["duration_min"] = duration              # минуты
    df_nodes["duration_estimated"] = duration_estimated  # True: в карточке нет длительности, взята медиана типа
    print(f"Цена: платных {int((df_nodes['price'] > 0).sum())}, бесплатных {int((df_nodes['price'] == 0).sum())}. "
          f"Длительность: оценена по медиане типа у {int(df_nodes['duration_estimated'].sum())}.")

    # 4. Валидация
    print("\nВалидация графа...")
    G = nx.from_pandas_edgelist(df_edges, source="source", target="target", edge_attr="weight", create_using=nx.DiGraph())

    is_dag = nx.is_directed_acyclic_graph(G)
    print(f"Граф является DAG?: {is_dag}")

    components = list(nx.weakly_connected_components(G))
    print(f"Компонент связности: {len(components)}")
    if len(components) > 0:
        largest_comp_size = len(max(components, key=len))
        print(f"Размер самой большой компоненты: {largest_comp_size} вершин (из {N})")

    # 5. Сохранение
    df_edges.to_parquet(OUT_EDGES_PATH, index=False)
    df_nodes.to_parquet(OUT_NODES_PATH, index=False)
    print(f"Сохранено: {OUT_EDGES_PATH}, {OUT_NODES_PATH}")

    # 6. Тест маршрута
    print("\n--- ТЕСТ МАРШРУТА (Налоги: от простого к сложному) ---")
    tax_mask = df_nodes["title"].str.contains("налог", case=False, na=False)
    if tax_mask.any():
        start_node = df_nodes[tax_mask]["difficulty"].idxmin()
        end_node = df_nodes[tax_mask]["difficulty"].idxmax()

        print(f"Старт: [{df_nodes.loc[start_node, 'difficulty']:.2f}] {df_nodes.loc[start_node, 'title']}")
        print(f"Финиш: [{df_nodes.loc[end_node, 'difficulty']:.2f}] {df_nodes.loc[end_node, 'title']}")

        try:
            # Инвертируем вес для Дейкстры (ищем максимальное сходство = минимальную стоимость)
            for u, v, d in G.edges(data=True):
                G[u][v]['cost'] = 1.0 - d['weight']

            path = nx.shortest_path(G, source=start_node, target=end_node, weight='cost')
            print(f"\nНайден маршрут ({len(path)} шагов):")
            for step, node_idx in enumerate(path):
                row = df_nodes.loc[node_idx]
                print(f"  {step + 1}. [{row['difficulty']:.2f}] {row['title']}  "
                      f"({row['duration_min']:.0f} мин, {row['price']:.0f} ₽)")
            route = df_nodes.loc[path]
            print(f"Итого по маршруту: {route['duration_min'].sum():.0f} мин, {route['price'].sum():.0f} ₽")
        except nx.NetworkXNoPath:
            print("\nПуть не найден (вершины в разных компонентах связности).")

if __name__ == "__main__":
    main()