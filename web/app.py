from flask import Flask, render_template, request, jsonify
import json

app = Flask(__name__)


@app.route('/')
def index():
    """Главная страница с формой"""
    return render_template('form.html')

@app.route('/submit', methods=['POST'])
def submit():
    """Обработка данных формы"""
    data = {
        'goal': {
            'direction': request.form.get('direction'),
            'description': request.form.get('description', '')
        },
        'currentState': {
            'level': request.form.get('level'),
            'knownSkills': request.form.getlist('skills'),  # множественный выбор
        },
        'constraints': {
            'weeklyHours': int(request.form.get('weeklyHours', 5)),
            'timeline': request.form.get('timeline'),
            'formats': request.form.getlist('formats')
        }
    }

    # Маппинг направлений из формы на читаемые русские названия
    DIRECTION_NAMES = {
        'start_strategy': 'узнать о старте и стратегии бизнеса',
        'finance_investments': 'разобраться в финансах и инвестициях',
        'tax_law': 'разобраться в налогах и праве в препринимательстве',
        'marketing_brand': 'узнать о маркетинге и построении бренда',
        'digital_marketing': 'узнать о маркетинге в интернете',
        'sales_clients': 'узнать больше о продажах и общении с клиентом',
        'analitics_data': 'обучиться аналитике данных',
        'team_hr': 'научиться собирать команду и работать с hr',
        'product_management': 'прокачать навыки менеджмента продукта',
        'product_dev': 'прокачать навыки разработки продукта',
        'tools': 'узнать о современных инструментах бизнеса',
    }

    # обрабатываем цель по-человечески
    goal_dict = data['goal']
    direction = goal_dict.get('direction', '')
    description = goal_dict.get('description', '').strip()

    if description:
        user_goal = description
    else:
        # Заглушка на основе направления
        readable_direction = [DIRECTION_NAMES.get(direction)]
        user_goal_dir = readable_direction[0]
        user_goal = f"Ваша цель — {user_goal_dir}"

    # Здесь будет вызов LLM (пока просто логируем)
    print("Получены данные:", json.dumps(data, ensure_ascii=False, indent=2))
    # вывод результата от модели - вывести хотим roadmap для пользователя
    mock_route = [
        {
            "course_title": "Выбор ниши и анализ рынка",
            "status": "completed",
            "time_hours": 4,
            "description": "Вы уже знаете основы, этот курс закрепляет материал."
        },
        {
            "course_title": "Регистрация бизнеса и налоги",
            "status": "current",
            "time_hours": 6,
            "description": "Ключевой этап. Узнайте, как выбрать ИП или ООО."
        },
        {
            "course_title": "Построение воронки продаж",
            "status": "locked",
            "time_hours": 8,
            "description": "Станет доступен после завершения предыдущего модуля."
        },
        {
            "course_title": "Масштабирование и управление командой",
            "status": "locked",
            "time_hours": 12,
            "description": "Финальная точка маршрута."
        }
    ]

    total_steps = len(mock_route)
    completed_steps = sum(1 for step in mock_route if step['status'] == 'completed')
    progress = int((completed_steps / total_steps) * 100)

    return render_template(
        'result.html',
        user_goal=user_goal,
        route=mock_route,
        progress=progress
    )

#    return render_template('result.html', data=data)

@app.route('/profile')
def profile():
    return render_template('profile.html')

if __name__ == '__main__':
    app.run(debug=True)