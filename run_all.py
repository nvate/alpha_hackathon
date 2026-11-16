import subprocess
import sys
import os

# Список скриптов для запуска (строго по порядку!)
PIPELINE_SCRIPTS = [
    "pipeline/align_datasets.py",
    "pipeline/build_tag_vectors.py",
    "pipeline/generate_difficulty.py",
    "pipeline/build_graph.py"
]


def run_script(script_path):
    print(f"\n{'=' * 50}\nЗАПУСК: {script_path}\n{'=' * 50}")
    result = subprocess.run([sys.executable, script_path], capture_output=False)
    if result.returncode != 0:
        print(f"ОШИБКА при выполнении {script_path}. Остановка.")
        sys.exit(1)


if __name__ == "__main__":
    print("🚀 НАЧАЛО СБОРКИ ДАННЫХ И ГРАФА...")

    # 1. Прогоняем пайплайн
    for script in PIPELINE_SCRIPTS:
        run_script(script)

    print("\n✅ ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН.")
    print("🌐 ЗАПУСК ВЕБ-СЕРВЕРА...\n")

    try:
        # Запускаем сервер и ждем, пока он работает
        subprocess.run([sys.executable, "web/app.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Сервер остановлен пользователем.")