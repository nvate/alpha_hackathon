from flask import Flask, render_template, request
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.form_parser import parse_form_data
from core.graph_router import build_route
from core.llm_explainer import generate_explanation, generate_step_commentary

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('form.html')

@app.route('/submit', methods=['POST'])
def submit():
    form_data = request.form.to_dict(flat=True)

    parsed_data = parse_form_data(form_data)

    route_steps = build_route(parsed_data)

    if not route_steps:
        route_steps = [
            {
                "course_title": "Материалы по вашей теме скоро появятся",
                "status": "current",
                "time_hours": 0,
                "description": "Попробуйте выбрать другое направление.",
                "difficulty": 0.0,
                "price": 0
            }
        ]

    for idx, step in enumerate(route_steps):
        step["description"] = generate_step_commentary(step, idx, len(route_steps))
        step["status"] = "current" if idx == 0 else "locked" # Статусы для таймлайна

    user_goal = form_data.get("description", "Персональный трек")
    overall_explanation = generate_explanation(route_steps, user_goal)

    total_steps = len(route_steps)
    progress = int((1 / total_steps) * 100) if total_steps > 0 else 0

    return render_template(
        'result.html',
        user_goal=user_goal,
        route=route_steps,
        progress=progress,
        overall_explanation=overall_explanation
    )

@app.route('/profile')
def profile():
    return render_template('profile.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)