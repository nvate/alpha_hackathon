import sys
import os
from flask import Flask, render_template, request

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
        return "Не удалось построить маршрут. Попробуйте изменить параметры формы.", 400

    for idx, step in enumerate(route_steps):
        step["description"] = generate_step_commentary(step, idx)

    user_goal = form_data.get("description", "Персональный трек")
    overall_explanation = generate_explanation(route_steps, user_goal)

    return render_template(
        'result.html',
        user_goal=user_goal,
        route=route_steps,
        progress=10,
        overall_explanation=overall_explanation
    )

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)