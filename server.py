import os, sys
from flask import Flask, render_template, request
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from sympy import symbols, lambdify, sympify


app = Flask(__name__)

# Constants for plotting and analysis
X_MIN = -10
X_MAX = 10
NUM_POINTS = 5000

# Function to analyze properties of a mathematical function
# Function to analyze properties of a mathematical function
def analyze_function(func_str):
    try:
        x = symbols('x')
        func = sympify(func_str)

        # Convert symbolic function to numerical
        f = lambdify(x, func, modules=['numpy'])

        # Create points array for analysis
        x_points = np.linspace(X_MIN, X_MAX, NUM_POINTS)
        y_points = f(x_points)

        # Find roots (where y is close to 0)
        roots = []
        for i in range(len(x_points)-1):
            if y_points[i] * y_points[i+1] <= 0:  # Sign change indicates root
                roots.append(round(float(x_points[i]), 2))

        # Find critical points and intervals by analyzing changes in y values
        critical_points = []
        growth_intervals = []
        decay_intervals = []

        current_direction = None
        interval_start = X_MIN

        for i in range(1, len(x_points)):
            diff = y_points[i] - y_points[i-1]
            
            # Determine if growing or decaying
            new_direction = 'growth' if diff > 0 else 'decay'
            
            # If direction changes, we found a critical point
            if current_direction and new_direction != current_direction:
                critical_points.append(round(float(x_points[i-1]), 2))
                
                # Record the interval that just ended
                interval = f"[{round(interval_start, 2)} , {round(float(x_points[i-1]), 2)}]"
                if current_direction == 'growth':
                    growth_intervals.append(interval)
                else:
                    decay_intervals.append(interval)
                    
                interval_start = x_points[i-1]
            
            current_direction = new_direction
        
        # Add the final interval
        final_interval = f"[{round(interval_start, 2)} , {round(float(X_MAX), 2)}]"
        if current_direction == 'growth':
            growth_intervals.append(final_interval)
        else:
            decay_intervals.append(final_interval)

        # Find the intersection point with the y-axis (x = 0)
        y_intercept = f(0)  # Calculate f(0)

        # Domain (Here it is assumed to be [-5, 5], but it can be calculated for more complex functions)
        domain = f"[{X_MIN} , {X_MAX}]"

        # Range (min and max values of the function)
        range_vals = f"[{round(np.min(y_points), 2)} , {round(np.max(y_points), 2)}]"

        return {
            "roots": roots if roots else "немає",
            "critical_points": critical_points if critical_points else "немає",
            "growth_intervals": " u ".join(growth_intervals) if growth_intervals else "немає",
            "decay_intervals": " u ".join(decay_intervals) if decay_intervals else "немає",
            "y_intercept": round(float(y_intercept), 2),  # Return y-intercept
            "domain": domain,
            "range": range_vals
        }

    except Exception as e:
        return {"error": f"Помилка при аналізі функції: {str(e)}"}




# Function to plot the graph of a mathematical function
def plot_function(func_str):
    try:
        x_sym = symbols('x')
        expr = sympify(func_str)
        
        # Explicitly specify numpy functions for lambdify
        func = lambdify(x_sym, expr, modules=[
            {'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 
             'sqrt': np.sqrt, 'log': np.log, 'exp': np.exp,
             'pi': np.pi},
            'numpy'
        ])
        
        # Using constants instead of hard-coded values
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
            config={
                'displayModeBar': False
            }
        )
    except Exception as e:
        return {"error": f"Помилка при побудові графіка: {str(e)}"}

@app.route('/')
def index():
    func_str = request.args.get('f', '')
    
    if func_str:
        try:
            plot_result = plot_function(func_str)
            
            # Check if we got an error from plot_function
            if isinstance(plot_result, dict) and 'error' in plot_result:
                return render_template('index.html', 
                                    error=plot_result['error'], 
                                    function=func_str)

            # If plot was successful, proceed with analysis
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
        app.run(host='0.0.0.0', debug=False)
    else:
        app.run(debug=True)
