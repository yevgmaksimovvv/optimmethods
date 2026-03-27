import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import threading

class OptimizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Методы оптимизации (по учебному пособию МАИ)")
        self.root.geometry("1400x800")
        
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.gradient_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.gradient_frame, text="Градиентный метод (раздел 4.3)")
        
        self.conjugate_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.conjugate_frame, text="Метод сопряженных градиентов (раздел 4.2)")
        
        self.setup_gradient_method()
        self.setup_conjugate_method()
    
    def safe_eval(self, expr, x1, x2):
        allowed_names = {
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
            'pi': np.pi, 'e': np.e,
            'x1': x1, 'x2': x2,
            'abs': abs, 'pow': pow
        }
        expr = expr.replace('^', '**')
        try:
            return eval(expr, {"__builtins__": {}}, allowed_names)
        except Exception as e:
            raise ValueError(f"Ошибка: {e}")
    
    def analytical_gradient(self, func_str, x1, x2):
        try:
            import sympy as sp
            x1_sym, x2_sym = sp.symbols('x1 x2')
            expr_str = func_str.replace('^', '**')
            f_sym = sp.sympify(expr_str)
            
            df_dx1 = sp.diff(f_sym, x1_sym)
            df_dx2 = sp.diff(f_sym, x2_sym)
            
            grad1 = float(df_dx1.subs({x1_sym: x1, x2_sym: x2}))
            grad2 = float(df_dx2.subs({x1_sym: x1, x2_sym: x2}))
            
            return np.array([grad1, grad2])
        except:
            return None
    
    def numerical_gradient(self, f, x1, x2, h=1e-6):
        grad = np.zeros(2)
        grad[0] = (f(x1 + h, x2) - f(x1 - h, x2)) / (2 * h)
        grad[1] = (f(x1, x2 + h) - f(x1, x2 - h)) / (2 * h)
        return grad
    
    def setup_gradient_method(self):
        left_frame = ttk.Frame(self.gradient_frame)
        left_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        
        func_frame = ttk.LabelFrame(left_frame, text="Функция для максимизации", padding=10)
        func_frame.pack(fill='x', pady=5)
        
        ttk.Label(func_frame, text="F(x1, x2) =").grid(row=0, column=0, sticky='w')
        self.grad_func_entry = ttk.Entry(func_frame, width=40)
        self.grad_func_entry.grid(row=0, column=1, padx=5, sticky='ew')
        self.grad_func_entry.insert(0, "x1**2 + x2**2 - x1*x2 + x1 - 2*x2")
        
        ttk.Label(func_frame, 
                 text="Задание: MAX F(x1,x2) = x1² + x2² - x1·x2 + x1 - 2x2, M0 = (0;0)",
                 foreground='blue').grid(row=1, column=0, columnspan=2, sticky='w', pady=5)
        
        params_frame = ttk.LabelFrame(left_frame, text="Параметры метода", padding=10)
        params_frame.pack(fill='x', pady=5)
        
        ttk.Label(params_frame, text="Начальная точка x1⁰:").grid(row=0, column=0, sticky='w')
        self.grad_x0 = ttk.Entry(params_frame, width=15)
        self.grad_x0.grid(row=0, column=1, padx=5)
        self.grad_x0.insert(0, "0")
        
        ttk.Label(params_frame, text="Начальная точка x2⁰:").grid(row=1, column=0, sticky='w')
        self.grad_y0 = ttk.Entry(params_frame, width=15)
        self.grad_y0.grid(row=1, column=1, padx=5)
        self.grad_y0.insert(0, "0")
        
        ttk.Label(params_frame, text="Начальный шаг h:").grid(row=2, column=0, sticky='w')
        self.grad_alpha = ttk.Entry(params_frame, width=15)
        self.grad_alpha.grid(row=2, column=1, padx=5)
        self.grad_alpha.insert(0, "0.1")
        
        ttk.Label(params_frame, text="Точность δ:").grid(row=3, column=0, sticky='w')
        self.grad_epsilon = ttk.Entry(params_frame, width=15)
        self.grad_epsilon.grid(row=3, column=1, padx=5)
        self.grad_epsilon.insert(0, "1e-6")
        
        ttk.Label(params_frame, text="Макс. итераций:").grid(row=4, column=0, sticky='w')
        self.grad_max_iter = ttk.Entry(params_frame, width=15)
        self.grad_max_iter.grid(row=4, column=1, padx=5)
        self.grad_max_iter.insert(0, "1000")
        
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill='x', pady=10)
        
        ttk.Button(btn_frame, text="Запустить оптимизацию", 
                  command=self.run_gradient_method).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Очистить вывод", 
                  command=lambda: self.clear_output(self.grad_output)).pack(side='left', padx=5)
        
        output_frame = ttk.LabelFrame(left_frame, text="Результаты", padding=10)
        output_frame.pack(fill='both', expand=True, pady=5)
        
        self.grad_output = scrolledtext.ScrolledText(output_frame, height=15, width=50)
        self.grad_output.pack(fill='both', expand=True)
        
        right_frame = ttk.Frame(self.gradient_frame)
        right_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        self.grad_fig = plt.Figure(figsize=(7, 6), dpi=100)
        self.grad_canvas = FigureCanvasTkAgg(self.grad_fig, right_frame)
        self.grad_canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def setup_conjugate_method(self):
        left_frame = ttk.Frame(self.conjugate_frame)
        left_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        
        func_frame = ttk.LabelFrame(left_frame, text="Функция для максимизации", padding=10)
        func_frame.pack(fill='x', pady=5)
        
        ttk.Label(func_frame, text="F(x1, x2) =").grid(row=0, column=0, sticky='w')
        self.conj_func_entry = ttk.Entry(func_frame, width=40)
        self.conj_func_entry.grid(row=0, column=1, padx=5, sticky='ew')
        self.conj_func_entry.insert(0, "-2 - x1 - 2*x2 - 0.1*x1**2 - 100*x2**2")
        
        ttk.Label(func_frame, 
                 text="Задание: MAX F(x1,x2) = -2 - x1 - 2x2 - 0.1x1² - 100x2², M0 = (1;1)",
                 foreground='blue').grid(row=1, column=0, columnspan=2, sticky='w', pady=5)
        
        params_frame = ttk.LabelFrame(left_frame, text="Параметры метода", padding=10)
        params_frame.pack(fill='x', pady=5)
        
        ttk.Label(params_frame, text="Начальная точка x1⁰:").grid(row=0, column=0, sticky='w')
        self.conj_x0 = ttk.Entry(params_frame, width=15)
        self.conj_x0.grid(row=0, column=1, padx=5)
        self.conj_x0.insert(0, "1")
        
        ttk.Label(params_frame, text="Начальная точка x2⁰:").grid(row=1, column=0, sticky='w')
        self.conj_y0 = ttk.Entry(params_frame, width=15)
        self.conj_y0.grid(row=1, column=1, padx=5)
        self.conj_y0.insert(0, "1")
        
        ttk.Label(params_frame, text="Точность ε:").grid(row=2, column=0, sticky='w')
        self.conj_epsilon = ttk.Entry(params_frame, width=15)
        self.conj_epsilon.grid(row=2, column=1, padx=5)
        self.conj_epsilon.insert(0, "1e-6")
        
        ttk.Label(params_frame, text="Макс. итераций:").grid(row=3, column=0, sticky='w')
        self.conj_max_iter = ttk.Entry(params_frame, width=15)
        self.conj_max_iter.grid(row=3, column=1, padx=5)
        self.conj_max_iter.insert(0, "100")
        
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill='x', pady=10)
        
        ttk.Button(btn_frame, text="Запустить оптимизацию", 
                  command=self.run_conjugate_method).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Очистить вывод", 
                  command=lambda: self.clear_output(self.conj_output)).pack(side='left', padx=5)
        
        output_frame = ttk.LabelFrame(left_frame, text="Результаты", padding=10)
        output_frame.pack(fill='both', expand=True, pady=5)
        
        self.conj_output = scrolledtext.ScrolledText(output_frame, height=15, width=50)
        self.conj_output.pack(fill='both', expand=True)
        
        right_frame = ttk.Frame(self.conjugate_frame)
        right_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        self.conj_fig = plt.Figure(figsize=(7, 6), dpi=100)
        self.conj_canvas = FigureCanvasTkAgg(self.conj_fig, right_frame)
        self.conj_canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def clear_output(self, output_widget):
        output_widget.delete(1.0, tk.END)
    
    def gradient_ascent(self, f, grad_f, x0, h0=0.1, delta=1e-6, max_iter=1000):
        """Градиентный метод для поиска стационарной точки"""
        x = np.array(x0, dtype=float)
        h = h0
        
        trajectory = [x.copy()]
        values = [f(x[0], x[1])]
        gradients = []
        steps = [h]
        
        iteration = 0
        step_reductions = 0
        
        for iteration in range(max_iter):
            grad = grad_f(x[0], x[1])
            gradients.append(grad.copy())
            
            if abs(grad[0]) <= delta and abs(grad[1]) <= delta:
                break
            
            x_trial = x - h * grad
            
            f_current = f(x[0], x[1])
            f_trial = f(x_trial[0], x_trial[1])
            
            if f_trial < f_current:
                x = x_trial
                trajectory.append(x.copy())
                values.append(f_trial)
                steps.append(h)
            else:
                h = h / 2
                step_reductions += 1
                continue
        
        return x, trajectory, values, gradients, steps, iteration + 1, step_reductions
    
    def conjugate_gradient_max(self, f, grad_f, x0, epsilon=1e-6, max_iter=100):
        """
        Метод сопряженных градиентов для максимизации квадратичной функции
        F = -2 - x1 - 2x2 - 0.1x1² - 100x2²
        """
        x = np.array(x0, dtype=float)
        
        grad = grad_f(x[0], x[1])
        s = grad.copy()
        
        trajectory = [x.copy()]
        values = [f(x[0], x[1])]
        gradients = [grad.copy()]
        
        for i in range(max_iter):
            grad_norm = np.linalg.norm(grad)
            if grad_norm < epsilon:
                break
            
            # Для квадратичной функции оптимальный шаг вычисляется аналитически
            # f(x + λs) = -2 - (x1+λs1) - 2(x2+λs2) - 0.1(x1+λs1)² - 100(x2+λs2)²
            # Это квадратичная функция по λ: a·λ² + b·λ + c
            # Максимум при λ = -b/(2a), где a < 0
            
            s1, s2 = s[0], s[1]
            x1, x2 = x[0], x[1]
            
            # Коэффициенты квадратичной функции
            a = -0.1 * s1**2 - 100 * s2**2
            b = -s1 - 2*s2 - 0.2 * x1 * s1 - 200 * x2 * s2
            c = -2 - x1 - 2*x2 - 0.1*x1**2 - 100*x2**2
            
            if a < 0:
                lam = -b / (2 * a)
                lam = max(0, min(lam, 2))  # Ограничиваем шаг
            else:
                lam = 0.1
            
            x_new = x + lam * s
            grad_new = grad_f(x_new[0], x_new[1])
            
            beta = np.dot(grad_new, grad_new) / (np.dot(grad, grad) + 1e-12)
            s_new = grad_new + beta * s
            
            x = x_new
            grad = grad_new
            s = s_new
            
            trajectory.append(x.copy())
            values.append(f(x[0], x[1]))
            gradients.append(grad.copy())
        
        return x, trajectory, values, gradients, i+1
    
    def run_gradient_method(self):
        def run():
            try:
                func_str = self.grad_func_entry.get()
                
                def f(x1, x2):
                    return self.safe_eval(func_str, x1, x2)
                
                grad_test = self.analytical_gradient(func_str, 0, 0)
                if grad_test is not None:
                    def grad_f(x1, x2):
                        return self.analytical_gradient(func_str, x1, x2)
                    grad_type = "аналитический"
                else:
                    def grad_f(x1, x2):
                        return self.numerical_gradient(f, x1, x2)
                    grad_type = "численный"
                
                x0 = float(self.grad_x0.get())
                y0 = float(self.grad_y0.get())
                h = float(self.grad_alpha.get())
                delta = float(self.grad_epsilon.get())
                max_iter = int(self.grad_max_iter.get())
                
                self.clear_output(self.grad_output)
                
                result, trajectory, values, gradients, steps, iterations, reductions = self.gradient_ascent(
                    f, grad_f, [x0, y0], h, delta, max_iter
                )
                
                output_text = "=" * 80 + "\n"
                output_text += "ГРАДИЕНТНЫЙ МЕТОД (раздел 4.3)\n"
                output_text += "Поиск стационарной точки (∇F = 0)\n"
                output_text += "=" * 80 + "\n\n"
                
                output_text += f"Функция: F(x1,x2) = {func_str}\n"
                output_text += f"Начальная точка M0: ({x0}, {y0})\n"
                output_text += f"F(M0) = {f(x0, y0):.6f}\n"
                output_text += f"Начальный шаг h = {h}\n"
                output_text += f"Точность δ = {delta}\n"
                output_text += f"Тип градиента: {grad_type}\n\n"
                
                # Аналитическое решение
                output_text += "Аналитическое решение (стационарная точка):\n"
                output_text += "  ∂F/∂x1 = 2x1 - x2 + 1 = 0\n"
                output_text += "  ∂F/∂x2 = 2x2 - x1 - 2 = 0\n"
                output_text += "  Решение: x1 = 0, x2 = 1\n"
                output_text += f"  F(0, 1) = {f(0, 1):.8f}\n\n"
                
                output_text += "-" * 90 + "\n"
                output_text += f"{'k':>4} | {'x1':>12} | {'x2':>12} | {'F(x)':>12} | {'∂F/∂x₁':>12} | {'∂F/∂x₂':>12} | {'h':>10}\n"
                output_text += "-" * 90 + "\n"
                
                for i in range(len(trajectory)):
                    point = trajectory[i]
                    val = values[i]
                    grad = gradients[i] if i < len(gradients) else gradients[-1]
                    step = steps[i] if i < len(steps) else steps[-1]
                    output_text += f"{i:>4} | {point[0]:>12.6f} | {point[1]:>12.6f} | {val:>12.6f} | {grad[0]:>12.6e} | {grad[1]:>12.6e} | {step:>10.6f}\n"
                
                output_text += "-" * 90 + "\n\n"
                
                output_text += f"Найденная стационарная точка:\n"
                output_text += f"  x1* = {result[0]:.8f}\n"
                output_text += f"  x2* = {result[1]:.8f}\n"
                output_text += f"  F* = {f(result[0], result[1]):.8f}\n\n"
                
                output_text += f"Статистика:\n"
                output_text += f"  Итераций: {iterations}\n"
                output_text += f"  Уменьшений шага: {reductions}\n"
                output_text += f"  Финальный шаг h: {steps[-1]:.6f}\n\n"
                
                final_grad = gradients[-1] if gradients else [0, 0]
                output_text += f"Проверка условия остановки (|∂F/∂x_j| ≤ δ):\n"
                output_text += f"  |∂F/∂x₁| = {abs(final_grad[0]):.2e} ≤ {delta} : {abs(final_grad[0]) <= delta}\n"
                output_text += f"  |∂F/∂x₂| = {abs(final_grad[1]):.2e} ≤ {delta} : {abs(final_grad[1]) <= delta}\n"
                
                self.grad_output.insert(tk.END, output_text)
                self.plot_gradient_results(trajectory, values, result, f, func_str)
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка: {str(e)}")
        
        threading.Thread(target=run, daemon=True).start()
    
    def run_conjugate_method(self):
        def run():
            try:
                func_str = self.conj_func_entry.get()
                
                def f(x1, x2):
                    return self.safe_eval(func_str, x1, x2)
                
                grad_test = self.analytical_gradient(func_str, 0, 0)
                if grad_test is not None:
                    def grad_f(x1, x2):
                        return self.analytical_gradient(func_str, x1, x2)
                    grad_type = "аналитический"
                else:
                    def grad_f(x1, x2):
                        return self.numerical_gradient(f, x1, x2)
                    grad_type = "численный"
                
                x0 = float(self.conj_x0.get())
                y0 = float(self.conj_y0.get())
                epsilon = float(self.conj_epsilon.get())
                max_iter = int(self.conj_max_iter.get())
                
                self.clear_output(self.conj_output)
                
                result, trajectory, values, gradients, iterations = self.conjugate_gradient_max(
                    f, grad_f, [x0, y0], epsilon, max_iter
                )
                
                output_text = "=" * 80 + "\n"
                output_text += "МЕТОД СОПРЯЖЕННЫХ ГРАДИЕНТОВ (раздел 4.2)\n"
                output_text += "Максимизация квадратичной функции\n"
                output_text += "=" * 80 + "\n\n"
                
                output_text += f"Функция: F(x1,x2) = {func_str}\n"
                output_text += f"Начальная точка M0: ({x0}, {y0})\n"
                output_text += f"F(M0) = {f(x0, y0):.6f}\n"
                output_text += f"Точность ε = {epsilon}\n"
                output_text += f"Тип градиента: {grad_type}\n\n"
                
                # Аналитическое решение
                output_text += "Аналитическое решение (максимум):\n"
                output_text += "  ∂F/∂x1 = -1 - 0.2x1 = 0 → x1 = -5\n"
                output_text += "  ∂F/∂x2 = -2 - 200x2 = 0 → x2 = -0.01\n"
                output_text += f"  F(-5, -0.01) = {f(-5, -0.01):.8f}\n\n"
                
                output_text += "-" * 80 + "\n"
                output_text += f"{'k':>4} | {'x1':>12} | {'x2':>12} | {'F(x)':>12} | {'||∇F||':>12}\n"
                output_text += "-" * 80 + "\n"
                
                for i, (point, val, grad) in enumerate(zip(trajectory, values, gradients)):
                    grad_norm = np.linalg.norm(grad)
                    output_text += f"{i:>4} | {point[0]:>12.6f} | {point[1]:>12.6f} | {val:>12.6f} | {grad_norm:>12.6e}\n"
                
                output_text += "-" * 80 + "\n\n"
                output_text += f"Найденный максимум:\n"
                output_text += f"  x1* = {result[0]:.8f}\n"
                output_text += f"  x2* = {result[1]:.8f}\n"
                output_text += f"  F* = {f(result[0], result[1]):.8f}\n"
                output_text += f"  Итераций: {iterations}\n"
                
                # Проверка точности
                true_x1, true_x2 = -5, -0.01
                error = np.sqrt((result[0] - true_x1)**2 + (result[1] - true_x2)**2)
                output_text += f"\nПогрешность относительно аналитического решения: {error:.2e}\n"
                
                self.conj_output.insert(tk.END, output_text)
                self.plot_conjugate_results(trajectory, values, result, f, func_str)
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка: {str(e)}")
        
        threading.Thread(target=run, daemon=True).start()
    
    def plot_gradient_results(self, trajectory, values, result, f, func_str):
        self.grad_fig.clear()
        
        ax1 = self.grad_fig.add_subplot(121, projection='3d')
        
        traj_x = [p[0] for p in trajectory]
        traj_y = [p[1] for p in trajectory]
        
        x_min = min(min(traj_x), result[0]) - 1
        x_max = max(max(traj_x), result[0]) + 1
        y_min = min(min(traj_y), result[1]) - 1
        y_max = max(max(traj_y), result[1]) + 1
        
        x = np.linspace(x_min, x_max, 50)
        y = np.linspace(y_min, y_max, 50)
        X, Y = np.meshgrid(x, y)
        
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    Z[i, j] = f(X[i, j], Y[i, j])
                except:
                    Z[i, j] = np.nan
        
        ax1.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis')
        
        traj_z = [f(p[0], p[1]) for p in trajectory]
        ax1.plot(traj_x, traj_y, traj_z, 'r.-', linewidth=2, markersize=8, label='Траектория')
        ax1.scatter([traj_x[0]], [traj_y[0]], [traj_z[0]], c='green', s=100, marker='o', label='M0')
        ax1.scatter([traj_x[-1]], [traj_y[-1]], [traj_z[-1]], c='red', s=100, marker='*', label='M*')
        
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        ax1.set_zlabel('F(x1,x2)')
        ax1.set_title('3D график функции')
        ax1.legend()
        
        ax2 = self.grad_fig.add_subplot(122)
        iterations = range(len(values))
        ax2.plot(iterations, values, 'b-', linewidth=2, marker='o', markersize=4)
        ax2.set_xlabel('Номер итерации k')
        ax2.set_ylabel('Значение F(x)')
        ax2.set_title('Сходимость метода')
        ax2.grid(True, alpha=0.3)
        
        self.grad_fig.tight_layout()
        self.grad_canvas.draw()
    
    def plot_conjugate_results(self, trajectory, values, result, f, func_str):
        self.conj_fig.clear()
        
        ax1 = self.conj_fig.add_subplot(121, projection='3d')
        
        traj_x = [p[0] for p in trajectory]
        traj_y = [p[1] for p in trajectory]
        
        x_min = -6
        x_max = 2
        y_min = -0.5
        y_max = 1.5
        
        x = np.linspace(x_min, x_max, 50)
        y = np.linspace(y_min, y_max, 50)
        X, Y = np.meshgrid(x, y)
        
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    Z[i, j] = f(X[i, j], Y[i, j])
                except:
                    Z[i, j] = np.nan
        
        ax1.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis')
        
        traj_z = [f(p[0], p[1]) for p in trajectory]
        ax1.plot(traj_x, traj_y, traj_z, 'r.-', linewidth=2, markersize=8, label='Траектория')
        ax1.scatter([traj_x[0]], [traj_y[0]], [traj_z[0]], c='green', s=100, marker='o', label='M0')
        ax1.scatter([traj_x[-1]], [traj_y[-1]], [traj_z[-1]], c='red', s=100, marker='*', label='M*')
        
        # Отметим аналитический максимум
        ax1.scatter([-5], [-0.01], [f(-5, -0.01)], c='yellow', s=100, marker='^', label='Аналитический max')
        
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        ax1.set_zlabel('F(x1,x2)')
        ax1.set_title('3D график функции (квадратичный максимум)')
        ax1.legend()
        
        ax2 = self.conj_fig.add_subplot(122)
        iterations = range(len(values))
        ax2.plot(iterations, values, 'b-', linewidth=2, marker='o', markersize=4)
        ax2.axhline(y=f(-5, -0.01), color='r', linestyle='--', label='Аналитический максимум')
        ax2.set_xlabel('Номер итерации k')
        ax2.set_ylabel('Значение F(x)')
        ax2.set_title('Сходимость метода сопряженных градиентов')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        self.conj_fig.tight_layout()
        self.conj_canvas.draw()

def main():
    root = tk.Tk()
    app = OptimizationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()