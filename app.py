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
            'hoursSpent': int(request.form.get('hoursSpent', 0))
        },
        'constraints': {
            'budget': request.form.get('budget'),
            'weeklyHours': int(request.form.get('weeklyHours', 5)),
            'timeline': request.form.get('timeline'),
            'formats': request.form.getlist('formats')
        }
    }

    # Здесь будет вызов LLM (пока просто логируем)
    print("Получены данные:", json.dumps(data, ensure_ascii=False, indent=2))
    # вывод результата от модели - вывести хотим roadmap для пользователя
    return render_template('result.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)