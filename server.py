import os
import sys
from typing import Dict, Union, List, Any, Tuple

import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from flask import Flask, render_template, request
from sympy import symbols, sympify, S, lambdify, And, diff, solve, limit, oo, solve_univariate_inequality, pretty, latex
from sympy.calculus.util import continuous_domain, function_range
import sympy

app = Flask(__name__)

# Налаштування для графіків: від -10 до 10 по осі X, використовуємо 5000 точок для плавності
X_MIN = -10
X_MAX = 10
NUM_POINTS = 5000

def get_domain_conditions(expr, x) -> List[Any]:
    """
    Знаходить умови для області визначення функції.
    Перевіряє:
    - знаменники (не можна ділити на нуль)
    - корені парного степеня (підкореневий вираз має бути невід'ємним)
    """
    domain_conditions = []
    
    # Перевіряємо знаменники (не можна ділити на нуль)
    for denom in expr.atoms(sympy.Pow):
        if denom.exp.is_negative:
            domain_conditions.append(denom.base != 0)
            
    # Перевіряємо корені парного степеня (підкореневий вираз має бути ≥ 0)
    for root in expr.atoms(sympy.Pow):
        if root.exp.is_Rational and root.exp.q == 2:
            domain_conditions.append(root.base >= 0)
            
    return domain_conditions

def get_range_str(expr, x) -> str:
    """
    Обчислює область значень функції.
    Повертає результат у вигляді рядка, наприклад:
    - "[0, ∞)" для x²
    - "(−∞, ∞)" для x
    """
    try:
        # Шукаємо мінімальне значення
        derivative = diff(expr, x)
        critical_points = solve(derivative, x)
        
        # Для sqrt(x) мінімум в точці x = 0
        min_val = float(expr.subs(x, 0))
        
        # Перевіряємо, чи функція обмежена зверху
        limit_inf = limit(expr, x, oo)
        
        if limit_inf == oo:
            return f"[{round(min_val, 2)}, ∞)"
        elif limit_inf == -oo:
            return f"(−∞, {round(max_val, 2)}]"
        else:
            max_val = float(limit_inf)
            return f"[{round(min_val, 2)}, {round(max_val, 2)}]"
            
    except Exception as e:
        return "немає"

def get_growth_decay_intervals(points, derivative, domain, x) -> Tuple[List[str], List[str]]:
    """Determine intervals where function grows or decays."""
    growth_intervals = []
    decay_intervals = []
    
    for i in range(len(points) - 1):
        mid_x = (points[i] + points[i + 1]) / 2
        try:
            if domain.subs(x, mid_x):
                deriv_val = float(derivative.subs(x, mid_x))
                interval = f"[{round(points[i], 2)}, {round(points[i + 1], 2)}]"
                if deriv_val > 0:
                    growth_intervals.append(interval)
                elif deriv_val < 0:
                    decay_intervals.append(interval)
        except:
            continue
            
    return growth_intervals, decay_intervals

def format_interval(interval) -> str:
    """
    Форматує математичні інтервали у зручний для читання вигляд.
    Замінює математичні символи на більш зрозумілі:
    - ∞ замість infinity
    - ≤ замість <=
    - ∪ для об'єднання множин
    тощо
    """
    if interval == S.EmptySet or interval == []:
        return "немає"
    if interval is False:
        return "немає"
    if interval is True:
        return "ℝ"

    # Конвертуємо в LaTeX і прибираємо зайві символи
    result = latex(interval)
    result = result.replace(r'\left', '')
    result = result.replace(r'\right', '')
    result = result.replace(r'\pi', 'π')
    result = result.replace(r'\cup', ' ∪ ')
    result = result.replace(r'\wedge', ' ∧ ')
    result = result.replace(r'\infty', '∞')
    result = result.replace(r'\leq', '≤')
    result = result.replace(r'\geq', '≥')
    result = result.replace(r'\vee', ' ∨ ')
    result = result.replace(r'\frac{', '')
    result = result.replace(r'\sqrt{', '√')
    result = result.replace('}{', '/')
    result = result.replace('}', '')
    result = result.replace('{', '')
    result = result.replace(r'\mathbb{R}', 'ℝ')
    result = result.replace('mathbbR', 'ℝ')
    result = result.replace('text{False}', 'немає')
    result = result.replace('textFalse', 'немає')
    result = result.replace('neq', '≠')
    # Прибираємо залишки LaTeX
    result = result.replace(r'\,', ' ')
    result = result.replace('\\', '')
    result = result.replace('[ ', '[')
    return result

def analyze_function(func_str: str, x_min: float = X_MIN, x_max: float = X_MAX) -> Dict[str, Any]:
    """
    Виконує повний математичний аналіз функції.
    
    Що саме аналізуємо:
    1. Область визначення (D) - всі допустимі x
    2. Область значень (E) - всі можливі y
    3. Нулі функції - де графік перетинає вісь X (тобто y = 0)
    4. Критичні точки - де функція "повертає" (вершини, ями)
    5. Проміжки зростання/спадання - де графік іде вгору/вниз
    6. Перетин з віссю Y - значення функції в точці x = 0
    
    Приклади:
    - f(x) = x² 
      - D: ℝ (всі числа)
      - E: [0, ∞) (від 0 до нескінченності)
      - нулі: x = 0 (одна точка)
      - критична точка: x = 0 (вершина параболи)
    
    - f(x) = 1/x
      - D: ℝ \ {0} (всі числа крім 0)
      - E: ℝ \ {0} (всі числа крім 0)
      - нулі: немає
      - критичних точок немає
    """
    try:
        # Створюємо символьну змінну x для математичних операцій
        x = symbols('x')
        # Перетворюємо текстовий запис функції на математичний вираз
        expr = sympify(func_str.replace("**", "^"))
        
        # Знаходимо область визначення та значень
        domain = continuous_domain(expr, x, S.Reals)  # де функція неперервна
        range_interval = function_range(expr, x, domain)  # які значення приймає
        # Збираємо всі результати аналізу
        
        roots = solve(expr, x)
        roots_str = ' = '.join([f"x = {round(float(root), 2)}" for root in roots]) if roots else "немає"
        critical_points = solve(diff(expr, x), x)
        critical_points_str = ' = '.join([f"x = {round(float(cp), 2)}" for cp in critical_points]) if critical_points else "немає"
        return {
            "domain": format_interval(domain),
            "range": format_interval(range_interval),
            # Розв'язуємо рівняння f(x) = 0
            "roots": roots_str,
            # Шукаємо точки, де похідна = 0 (вершини, ями)
            "critical_points": critical_points_str,
            # Де похідна > 0, там функція зростає
            "growth_intervals": format_interval(solve_univariate_inequality(diff(expr, x) >= 0, x)) or "немає",  
            "decay_intervals": format_interval(solve_univariate_inequality(diff(expr, x) <= 0, x)),
            # Перевіряємо чи x = 0 входить в область визначення
            "y_intercept": f"y = {float(expr.subs(x, 0))}" if 0 in domain else "немає"
        }
        
    except Exception as e:
        return {"error": f"Помилка при аналізі функції: {str(e)}"}

def plot_function(func_str: str, x_min: float = X_MIN, x_max: float = X_MAX) -> Union[str, Dict[str, str]]:
    """
    Створює інтерактивний графік функції.
    
    Як це працює:
    1. Беремо проміжок [x_min, x_max] і ділимо його на NUM_POINTS точок
    2. Обчислюємо значення функції в кожній точці
    3. З'єднуємо точки плавною лінією
    
    Особливості:
    - Використовуємо багато точок (5000) щоб графік був гладким
    - Автоматично масштабуємо осі щоб було видно всі важливі частини графіка
    - Можна наводити мишкою і бачити координати точок
    
    Підтримує різні типи функцій:
    - Многочлени: x², x³ + 2x - 1
    - Тригонометричні: sin(x), cos(x), tan(x)
    - Корені: sqrt(x), cbrt(x)
    - Експоненти та логарифми: exp(x), log(x)
    - Дроби: 1/x, (x²+1)/(x-1)
    """
    # Замінюємо ** на ^ бо так звичніше записувати степені
    func_str = func_str.replace("**", "^")
    
    try:
        # Підготовка функції до обчислень
        x_sym = symbols('x')  # символьна змінна для SymPy
        expr = sympify(func_str)  # перетворюємо текст на математичний вираз
        
        # Створюємо функцію, яку можна обчислити
        # Додаємо всі математичні функції які можуть знадобитися
        func = lambdify(x_sym, expr, modules=[
            {'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
             'sqrt': np.sqrt, 'log': np.log, 'exp': np.exp,
             'pi': np.pi},  # математичні функції
            'numpy'  # для швидких обчислень
        ])
        
        # Створюємо масив точок для графіка
        x = np.linspace(x_min, x_max, NUM_POINTS)  # рівномірно розподілені точки
        y = func(x)  # обчислюємо значення функції в кожній точці
        
        # Створюємо графік
        fig = go.Figure()
        # Додаємо лінію графіка
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',  # режим відображення - неперервна лінія
            name='f(x)'    # підпис для легенди
        ))
        
        # Налаштовуємо зовнішній вигляд
        fig.update_layout(
            title=dict(
                text=f'f(x) = {func_str}',  # заголовок графіка
                x=0.5  # центруємо заголовок
            )
        )
        
        # Конвертуємо графік в HTML для відображення на сайті
        return pio.to_html(
            fig,
            full_html=False,  # повертаємо тільки частину з графіком
            config={'displayModeBar': False}  # прибираємо зайві кнопки
        )
    except Exception as e:
        return {"error": f"Помилка при побудові графіка: {str(e)}"}

@app.route('/')
def index():
    """
    Головна сторінка програми.
    Показує форму для введення функції та результати аналізу.
    """
    # Отримуємо параметри з URL
    func_str = request.args.get('f', '')  # функція
    x_min = float(request.args.get('xmin', X_MIN))  # ліва межа
    x_max = float(request.args.get('xmax', X_MAX))  # права межа
    
    # Якщо функція не задана, показуємо пусту форму
    if not func_str:
        return render_template('index.html', 
                             function=func_str,
                             x_min=x_min,
                             x_max=x_max)
    
    try:
        # Спробуємо намалювати графік
        plot_result = plot_function(func_str, x_min, x_max)
        
        # Якщо виникла помилка при побудові графіка
        if isinstance(plot_result, dict) and 'error' in plot_result:
            return render_template('index.html', 
                                 error=plot_result['error'], 
                                 function=func_str,
                                 x_min=x_min,
                                 x_max=x_max)
        
        # Аналізуємо функцію та показуємо результати
        analysis = analyze_function(func_str, x_min, x_max)
        return render_template('index.html', 
                             plot_html=plot_result, 
                             analysis=analysis, 
                             function=func_str,
                             x_min=x_min,
                             x_max=x_max)
        
    except Exception as e:
        return render_template('index.html', 
                             error=f"Несподівана помилка: {str(e)}", 
                             function=func_str,
                             x_min=x_min,
                             x_max=x_max)

# Запускаємо програму
if __name__ == '__main__':
    if '--prod' in sys.argv:  # якщо запускаємо на сервері
        app.run(
            host='0.0.0.0',
            port=int(os.environ.get('PORT', 5000)),
            debug=False
        )
    else:  # якщо запускаємо для розробки
        app.run(debug=True, port=5001)


