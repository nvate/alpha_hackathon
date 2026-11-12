import numpy as np
import pandas as pd
from pathlib import Path
import networkx as nx

# --- Пути к артефактам ---
VECTORS_PATH = Path("artifacts/tag_vectors.npy")
DIFFICULTY_PATH = Path("artifacts/difficulty_scores.npy")
META_PATH = Path("artifacts/materials_meta.parquet")

OUT_EDGES_PATH = Path("artifacts/graph_edges.parquet")
OUT_NODES_PATH = Path("artifacts/graph_nodes.parquet")

# --- Параметры графа ---
TOP_K = 20  # Увеличили: смотрим на 20 ближайших соседей
MIN_SIMILARITY = 0.15  # Снизили порог, чтобы связать больше статей


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
    df_nodes["price"] = 0.0
    df_nodes["duration_min"] = 10.0

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
                print(f"  {step + 1}. [{row['difficulty']:.2f}] {row['title']}")
        except nx.NetworkXNoPath:
            print("\nПуть не найден (вершины в разных компонентах связности).")

if __name__ == "__main__":
    main()