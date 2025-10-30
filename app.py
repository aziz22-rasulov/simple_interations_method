import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math

# Настройка страницы
st.set_page_config(page_title="Метод простой итерации", layout="wide")
st.title("📈 Визуализация метода простой итерации")

# Боковая панель для настроек
with st.sidebar:
    st.header("Настройки решения")
    g_str = st.text_input("Функция g(x)", "2 - math.cos(x)")
    st.caption("Примеры: '2 - math.cos(x)', '(x + 2/x)/2', 'math.sqrt(2)'")
    
    x0 = st.number_input("Начальное приближение x0", value=2.98, step=0.1)
    epsilon = st.number_input("Точность (epsilon)", value=1e-6, format="%.6f")
    max_iter = st.slider("Максимальное число итераций", 10, 1000, 100)
    
    st.markdown("---")
    st.info("""
    **Важно!** 
    - Преобразуйте уравнение f(x)=0 в вид x = g(x)
    - Используйте math. для математических функций
    - Для корня x²=2 используйте g(x)=(x+2/x)/2
    """)

# Кнопка запуска
if st.sidebar.button("Рассчитать"):
    try:
        # Создаем функцию g(x) из строки
        g = lambda x: eval(g_str, {"math": math, "np": np}, {"x": x})
        
        # Подготовка данных для графика
        # Диапазон: x0 ± 3, но с умным смещением
        x_min = max(0, x0 - 3)
        x_max = x0 + 3
        x_vals = np.linspace(x_min, x_max, 1000)
        y_g = [g(x) for x in x_vals]
        y_x = x_vals  # прямая y = x
        
        # Вычисление итераций
        x_prev = x0
        iterations = [x0]
        for i in range(max_iter):
            x_next = g(x_prev)
            iterations.append(x_next)
            if abs(x_next - x_prev) < epsilon:
                break
            x_prev = x_next
        
        # Построение графика
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # График g(x) и y=x
        ax.plot(x_vals, y_g, label='y = g(x)', color='#1f77b4', linewidth=2)
        ax.plot(x_vals, y_x, label='y = x', color='#ff7f0e', linestyle='--', linewidth=2)
        
        # Лестничная диаграмма (итерации)
        for i in range(len(iterations)-1):
            # Горизонтальный отрезок (x_n -> x_{n+1} на y=x)
            ax.plot([iterations[i], iterations[i+1]], [iterations[i], iterations[i]], 
                    color='#2ca02c', alpha=0.8, linewidth=1.5)
            # Вертикальный отрезок (x_{n+1} -> y=g(x_{n+1}))
            ax.plot([iterations[i+1], iterations[i+1]], [iterations[i], iterations[i+1]], 
                    color='#2ca02c', alpha=0.8, linewidth=1.5)
        
        # Точки итераций
        ax.scatter(iterations, iterations, color='#d62728', s=50, zorder=5)
        
        # Подписи
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title('Процесс сходимости метода простой итерации', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.1)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.1)
        
        # Отображение графика
        st.pyplot(fig)
        
        # Вывод результатов
        st.subheader("✅ Результаты вычислений")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Корень уравнения", f"{iterations[-1]:.8f}")
            st.metric("Число итераций", len(iterations)-1)
            st.metric("Точность", f"{abs(iterations[-1] - iterations[-2]):.2e}")
        
        with col2:
            st.metric("Проверка g(x)", f"{g(iterations[-1]):.8f}")
            st.metric("Разница |x - g(x)|", f"{abs(iterations[-1] - g(iterations[-1])):.2e}")
            st.metric("Сходимость", "✅ Сходится" if len(iterations) < max_iter else "⚠️ Достигнуто max_iter")
        
        # Дополнительная информация
        st.markdown("### 📝 Как читать график:")
        st.markdown("""
        - **Синяя линия**: график функции y = g(x)
        - **Оранжевая пунктирная линия**: прямая y = x
        - **Зелёные линии**: шаги метода итерации
        - **Красные точки**: значения на каждой итерации
        """)
        
        # Визуализация сходимости
        if len(iterations) >= max_iter:
            st.warning("❗ Метод не сходится за заданное число итераций! Попробуйте:")
            st.markdown("""
            - Изменить начальное приближение
            - Выбрать другое преобразование x = g(x)
            - Увеличить максимальное число итераций
            """)
            
        # Проверка производной для сходимости
        try:
            h = 1e-5
            x_check = iterations[-1]
            deriv = (g(x_check + h) - g(x_check - h)) / (2 * h)
            if abs(deriv) >= 1:
                st.error(f"⚠️ Внимание: |g'(x)| = {abs(deriv):.4f} ≥ 1 в точке корня. Метод может не сходиться!")
            else:
                st.success(f"✅ Условие сходимости: |g'(x)| = {abs(deriv):.4f} < 1")
        except:
            pass
            
    except Exception as e:
        st.error(f"❌ Ошибка в вычислениях: {str(e)}")
        st.markdown("""
        **Проверьте:**
        - Корректность функции g(x)
        - Начальное приближение
        - Ограничения функции (например, arccos от числа >1)
        """)
        st.code("""
        Примеры корректных функций:
        - Для x + cos(x) = 2: 2 - math.cos(x)
        - Для x² - 2 = 0: (x + 2/x)/2
        - Для x³ - x - 1 = 0: (x + 1)**(1/3)
        """)

# Инструкция для пользователя
st.markdown("""
---
### 📚 Как использовать это приложение:
1. В поле **"Функция g(x)"** введите преобразованное уравнение в виде x = g(x)
2. Задайте **начальное приближение x0**
3. Настройте **точность** и **максимальное число итераций**
4. Нажмите **"Рассчитать"**
5. Наблюдайте за графиком и результатами!

> 💡 **Совет**: Если метод не сходится, попробуйте:
> - Изменить начальное приближение
> - Выбрать другое преобразование уравнения
> - Увеличить число итераций
""")