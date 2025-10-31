import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math

# Функция для поиска интервалов с корнями
def find_intervals(f, x_min, x_max, step=0.1):
    intervals = []
    x = x_min
    while x < x_max:
        try:
            f1 = f(x)
            f2 = f(x + step)
            if f1 * f2 < 0:
                intervals.append((x, x + step))
        except: pass
        x += step
    return intervals

# Функция для численного дифференцирования
def numerical_derivative(f, x, h=1e-5):
    try: return (f(x + h) - f(x - h)) / (2 * h)
    except: return None

# Собственная реализация метода простой итерации
def simple_iteration(g, x0, epsilon, max_iter):
    x_prev = x0
    iterations = [x0]
    errors = []
    
    for _ in range(max_iter):
        try:
            x_next = g(x_prev)
        except Exception as e:
            return None, None, f"Ошибка: {str(e)}"
        
        error = abs(x_next - x_prev)
        errors.append(error)
        iterations.append(x_next)
        
        if error < epsilon:
            break
            
        x_prev = x_next
    
    return iterations, errors, None

# Настройка страницы
st.set_page_config(page_title="Метод простой итерации", layout="wide")
st.title("🔍 Решение нелинейных уравнений (только метод простой итерации)")

# Боковая панель
with st.sidebar:
    st.header("Ввод уравнения")
    
    # Поле для f(x) = 0 в Python-формате
    f_str = st.text_input("f(x) = 0", value="x + math.cos(x) - 2")
    st.caption("Примеры: x + math.cos(x) - 2, x**2 - 2, math.exp(x) - x - 2")
    
    # Поле для g(x) = x
    g_str = st.text_input("Функция g(x) (x = g(x))", value="2 - math.cos(x)")
    st.caption("Примеры: 2 - math.cos(x), (x + 2/x)/2, math.log(x + 2)")
    
    x0 = st.number_input("Начальное приближение x0", 2.98, step=0.1)
    epsilon = st.number_input("Точность", 1e-6, format="%.6f")
    max_iter = st.slider("Макс. итераций", 10, 1000, 100)
    
    col1, col2 = st.columns(2)
    with col1: find_intervals_btn = st.button("Найти интервалы")
    with col2: study_convergence_btn = st.button("Исследовать скорость сходимости")
    
    st.info("Важно: Для метода простой итерации важно правильно преобразовать уравнение в вид x = g(x)")

# Кнопка "Рассчитать"
if st.sidebar.button("Рассчитать"):
    try:
        # Создаем функции f(x) и g(x)
        f = lambda x: eval(f_str, {"math": math, "np": np}, {"x": x})
        g = lambda x: eval(g_str, {"math": math, "np": np}, {"x": x})
        
        # Поиск интервалов
        intervals = find_intervals(f, x0-5, x0+5)
        
        # Выполняем метод простой итерации
        iterations, errors, error_msg = simple_iteration(g, x0, epsilon, max_iter)
        if error_msg: raise Exception(error_msg)
        
        # Проверка условия сходимости
        deriv = numerical_derivative(g, iterations[-1])
        condition = "✅ Сходится" if deriv is not None and abs(deriv) < 1 else "⚠️ Не сходится"
        
        # Вывод результатов
        st.subheader("✅ Результаты вычислений")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Корень", f"{iterations[-1]:.8f}")
            st.metric("Итераций", len(iterations)-1)
            st.metric("Точность", f"{abs(iterations[-1]-iterations[-2]):.2e}")
        with col2:
            st.metric("g(x)", f"{g(iterations[-1]):.8f}")
            st.metric("|x - g(x)|", f"{abs(iterations[-1]-g(iterations[-1])):.2e}")
            st.metric("Сходимость", condition)
        
        # Интервалы
        st.subheader("🔍 Интервалы с корнями")
        if intervals:
            for i, (a, b) in enumerate(intervals):
                st.write(f"Интервал {i+1}: [{a:.4f}, {b:.4f}]")
        else:
            st.warning("Интервалы не найдены")
        
        # График процесса сходимости
        st.subheader("📊 График процесса сходимости")
        fig, ax = plt.subplots(figsize=(10, 6))
        x_min, x_max = min(iterations)-1, max(iterations)+1
        x_vals = np.linspace(x_min, x_max, 1000)
        y_g = [g(x) for x in x_vals]
        y_x = x_vals
        
        ax.plot(x_vals, y_g, 'b-', label='y = g(x)')
        ax.plot(x_vals, y_x, 'r--', label='y = x')
        
        for i in range(len(iterations)-1):
            ax.plot([iterations[i], iterations[i+1]], [iterations[i], iterations[i]], 'g-', alpha=0.8)
            ax.plot([iterations[i+1], iterations[i+1]], [iterations[i], iterations[i+1]], 'g-', alpha=0.8)
        
        ax.scatter(iterations, iterations, color='red', s=50)
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.grid(True); ax.legend()
        st.pyplot(fig)
        
        # График скорости сходимости
        st.subheader("📈 График скорости сходимости")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(range(len(errors)), np.log10(errors), 'bo-')
        ax2.set_xlabel('Номер итерации'); ax2.set_ylabel('log10(ошибка)'); ax2.grid(True)
        st.pyplot(fig2)
        
        # Исследование скорости от точности
        if study_convergence_btn:
            st.subheader("📊 Зависимость итераций от точности")
            epsilons = [10**(-i) for i in range(2, 11)]
            iters = []
            for eps in epsilons:
                _, _, errors, _ = simple_iteration(g, x0, eps, max_iter)
                iters.append(len(errors))
            
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            ax3.plot(np.log10(epsilons), iters, 'go-')
            ax3.set_xlabel('log10(точность)'); ax3.set_ylabel('Число итераций'); ax3.grid(True)
            st.pyplot(fig3)
    
    except Exception as e:
        st.error(f"❌ Ошибка: {str(e)}")
        st.markdown("""
        **Проверьте:**
        - Корректность функции g(x)
        - Начальное приближение
        - Ограничения функции (например, arccos от числа >1)
        """)
        st.code("""
        Примеры корректных функций:
        - Для x + cos(x) = 2: 2 - math.cos(x)
        - Для x^2 - 2 = 0: (x + 2/x)/2
        - Для x^3 - x - 1 = 0: (x + 1)**(1/3)
        """)
