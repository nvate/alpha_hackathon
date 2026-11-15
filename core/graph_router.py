import networkx as nx
import pandas as pd
from pathlib import Path

NODES_PATH = Path("artifacts/graph_nodes.parquet")
EDGES_PATH = Path("artifacts/graph_edges.parquet")

print("Загружаю граф...")
nodes_df = pd.read_parquet(NODES_PATH)
edges_df = pd.read_parquet(EDGES_PATH)

G = nx.from_pandas_edgelist(
    edges_df, source="source", target="target",
    edge_attr="weight", create_using=nx.DiGraph()
)


def build_route(parsed_data: dict) -> list:
    target_tag = parsed_data["target_tag"]
    user_diff = parsed_data["user_difficulty"]

    # 1. Фильтруем вершины ТОЛЬКО по теме (никакого фильтра по цене!)
    topic_keywords = {
        "налоги_и_законы": "налог",
        "финансы_и_учет": "финанс",
        "маркетинг": "маркетинг",
        "старт_бизнеса": "бизнес",
        "продажи_и_переговоры": "продаж",
        "управление_командой": "команд",
        "анализ_данных": "анализ данных",
        "технологии": "ии"
    }
    keyword = topic_keywords.get(target_tag, "бизнес")

    mask = nodes_df["title"].str.contains(keyword, case=False, na=False)
    topic_nodes = nodes_df[mask]

    if topic_nodes.empty:
        return []

    # 2. Стартовая нода: ближайшая по сложности к пользователю (ищем среди бесплатных, чтобы не спугнуть)
    free_topic_nodes = topic_nodes[topic_nodes["price"] == 0]
    if not free_topic_nodes.empty:
        start_node = (free_topic_nodes["difficulty"] - user_diff).abs().idxmin()
    else:
        start_node = (topic_nodes["difficulty"] - user_diff).abs().idxmin()

    # 3. Конечная нода: САМАЯ СЛОЖНАЯ, и приоритет — платным/подпискам (Monetization Goal)
    # Ищем самые сложные материалы (топ 30%), из них выбираем самый дорогой (или подписку)
    top_difficulty_nodes = topic_nodes[topic_nodes["difficulty"] > topic_nodes["difficulty"].quantile(0.7)]

    if not top_difficulty_nodes.empty:
        # Если есть платные/подписки на верхушке сложности — ведем именно туда!
        paid_top = top_difficulty_nodes[top_difficulty_nodes["price"] > 0]
        if not paid_top.empty:
            end_node = paid_top["price"].idxmax()  # Ведем к самому дорогому
        else:
            end_node = top_difficulty_nodes["difficulty"].idxmax()
    else:
        end_node = topic_nodes["difficulty"].idxmax()

    # 4. Динамическое взвешивание (Multi-Objective Optimization)
    # Штраф за цену на ранних этапах, бонус за цену на поздних
    for u, v, d in G.edges(data=True):
        base_cost = 1.0 - d["weight"]

        # Бизнес-логика: Если курс платный, мы штрафуем его на старте пути
        # (чтобы пользователь сначала прокачался на бесплатном),
        # но поощряем ближе к концу, чтобы была конверсия.
        target_price = nodes_df.loc[v, "price"]

        if target_price > 0:
            # Эвристика: если сложность узла высокая (мы уже далеко в пути), цена не мешает
            if nodes_df.loc[v, "difficulty"] > 0.5:
                base_cost -= 0.1  # Скидка (алгоритму выгоднее вести сюда)
            else:
                base_cost += 0.5  # Штраф (не продаем на первых шагах)

        G[u][v]["cost"] = base_cost

    # 5. Строим путь
    try:
        path = nx.shortest_path(G, source=start_node, target=end_node, weight="cost")
    except nx.NetworkXNoPath:
        return []

    # 6. Формируем красивый список
    route_steps = []
    for step_idx, node_id in enumerate(path):
        row = nodes_df.loc[node_id]

        # В MVP первый шаг current, остальные locked
        status = "current" if step_idx == 0 else "locked"

        route_steps.append({
            "id": int(node_id),
            "course_title": row["title"],
            "difficulty": float(row["difficulty"]),
            "description": "",  # Заполнится LLM-ом
            "time_hours": round(row["duration_min"] / 60, 1),
            "price": float(row["price"]),
            "status": status
        })

    return route_steps