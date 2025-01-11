import os
import sys
from flask import Flask, render_template, request
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from sympy import symbols, sympify, S, lambdify, And
import sympy

app = Flask(__name__)

# Constants for plotting and analysis
X_MIN = -10
X_MAX = 10
NUM_POINTS = 5000


def analyze_function(func_str):
    func_str = func_str.replace("**", "^")
    try:
        x = symbols('x')
        func = sympify(func_str)

        # Перевірка області визначення вручну
        domain_conditions = []

        # Для коренів
        roots = func.atoms(sympy.Pow)
        for r in roots:
            if r.exp.is_Rational and r.exp.q == 2:  # Перевірка кореня парного ступеня
                domain_conditions.append(r.base >= 0)

        # Для дробів
        fractions = func.atoms(sympy.Mul)
        for f in fractions:
            if f.is_Rational:
                denom = sympy.denom(f)
                domain_conditions.append(denom != 0)

        # Об'єднуємо всі умови
        domain = And(*domain_conditions) if domain_conditions else S.Reals

        # Перетворення області визначення у людський формат
        domain_str = str(domain) if domain != S.Reals else "x ∈ R"

        # Перетворення символічної функції в числову
        f = lambdify(x, func, modules=['numpy'])

        # Створення точок для аналізу
        x_points = np.linspace(X_MIN, X_MAX, NUM_POINTS)
        y_points = []

        # Виключення точок, які не належать області визначення
        for x_val in x_points:
            if domain.subs(x, x_val):
                try:
                    y_points.append(f(x_val))
                except Exception:
                    y_points.append(np.nan)
            else:
                y_points.append(np.nan)

        y_points = np.array(y_points)

        # Знаходження коренів функції (перетини з віссю OX)
        roots = []
        for i in range(len(x_points)-1):
            if not np.isnan(y_points[i]) and not np.isnan(y_points[i+1]):
                if y_points[i] * y_points[i+1] <= 0:  # Зміна знаку вказує на корінь
                    roots.append(round(float(x_points[i]), 2))

        # Аналіз критичних точок та інтервалів зростання/спадання
        critical_xmin = []  # Точки мінімуму
        critical_xmax = []  # Точки максимуму
        growth_intervals = []
        decay_intervals = []

        current_direction = None
        interval_start = X_MIN

        for i in range(1, len(x_points)):
            if not np.isnan(y_points[i]) and not np.isnan(y_points[i-1]):
                diff = y_points[i] - y_points[i-1]

                # Визначення напрямку (зростання чи спадання)
                new_direction = 'growth' if diff > 0 else 'decay'

                # Якщо напрямок змінився, це критична точка
                if current_direction and new_direction != current_direction:
                    critical_xmin.append(round(float(x_points[i-1]), 2)) if current_direction == 'decay' else None
                    critical_xmax.append(round(float(x_points[i-1]), 2)) if current_direction == 'growth' else None

                    interval = f"[{round(interval_start, 2)} , {round(float(x_points[i-1]), 2)}]"
                    if current_direction == 'growth':
                        growth_intervals.append(interval)
                    else:
                        decay_intervals.append(interval)

                    interval_start = x_points[i-1]

                current_direction = new_direction

        # Додавання останнього інтервалу
        final_interval = f"[{round(interval_start, 2)} , {round(float(X_MAX), 2)}]"
        if current_direction == 'growth':
            growth_intervals.append(final_interval)
        else:
            decay_intervals.append(final_interval)

        # Перетин з віссю OY (x = 0)
        y_intercept = f(0) if domain.subs(x, 0) else "немає"
        y_intercept = f"y = {y_intercept}"  


        # Діапазон функції (мінімум та максимум)
        min_val = round(np.nanmin(y_points), 2) if len(y_points) > 0 else "немає"
        max_val = round(np.nanmax(y_points), 2) if len(y_points) > 0 else "немає"

        range_str = f"[{min_val}, {max_val}]"


        # Критичні точки у вигляді рядка
        critical_points_combined = ", ".join(
    map(lambda x: str(int(x)) if x.is_integer() else str(x), critical_xmin + critical_xmax)
) if critical_xmin or critical_xmax else "немає"


        # Корені у вигляді рядка
        roots_str = f"x = {', '.join(map(str, roots))}" if roots else "немає"

        return {
            "roots": roots_str,
            "critical_points": critical_points_combined,
            "growth_intervals": " ∪ ".join(growth_intervals) if growth_intervals else "немає",
            "decay_intervals": " ∪ ".join(decay_intervals) if decay_intervals else "немає",
            "y_intercept": y_intercept,
            "domain": domain_str,
            "range": range_str,
            "min_points": critical_xmin,  # Додайте точки мінімуму
            "max_points": critical_xmax  # Додайте точки максимуму
}


    except Exception as e:
        return {"error": f"Помилка при аналізі функції: {str(e)}"}




# Function to plot the graph of a mathematical function
def plot_function(func_str):
    func_str = func_str.replace("**", "^")
    try:
        x_sym = symbols('x')
        expr = sympify(func_str)

        func = lambdify(x_sym, expr, modules=[
            {'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
             'sqrt': np.sqrt, 'log': np.log, 'exp': np.exp,
             'pi': np.pi},
            'numpy'
        ])

        x = np.linspace(X_MIN, X_MAX, NUM_POINTS)
        y = func(x)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))
        fig.update_layout(
            title=dict(
                text=f'f(x) = {func_str}',
                x=0.5
            )
        )

        return pio.to_html(
            fig,
            full_html=False,
            config={'displayModeBar': False}
        )
    except Exception as e:
        return {"error": f"Помилка при побудові графіка: {str(e)}"}


@app.route('/')
def index():
    func_str = request.args.get('f', '')

    if func_str:
        try:
            plot_result = plot_function(func_str)

            if isinstance(plot_result, dict) and 'error' in plot_result:
                return render_template('index.html', 
                                       error=plot_result['error'], 
                                       function=func_str)

            analysis = analyze_function(func_str)
            return render_template('index.html', 
                                   plot_html=plot_result, 
                                   analysis=analysis, 
                                   function=func_str)

        except Exception as e:
            return render_template('index.html', 
                                   error=f"Несподівана помилка: {str(e)}", 
                                   function=func_str)

    return render_template('index.html', function=func_str)

if __name__ == '__main__':
    if '--prod' in sys.argv:
        app.run(
            host='0.0.0.0',
            port=int(os.environ.get('PORT', 5000)),
            debug=False
        )
    else:
        app.run(debug=True)

