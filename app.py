import streamlit as st
import numpy as np
import time
from numpy.linalg import norm

st.set_page_config(page_title="Схема Халецкого", page_icon="🧮", layout="wide")


def haltsky_decomposition(A):
    """Разложение A = B*C по формулам из учебника"""
    n = len(A)
    B = np.zeros((n, n))
    C = np.zeros((n, n))
    
    for i in range(n):
        C[i, i] = 1.0
    
    for j in range(n):
        for i in range(j, n):
            if j == 0:
                B[i, j] = A[i, j]
            else:
                sum_val = 0.0
                for k in range(j):
                    sum_val += B[i, k] * C[k, j]
                B[i, j] = A[i, j] - sum_val
            
            if i == j and abs(B[i, j]) < 1e-10:
                raise ValueError(f"Элемент B[{i+1}][{j+1}] = {B[i, j]:.4e} близок к нулю")
        
        for i in range(j+1, n):
            sum_val = 0.0
            for k in range(j):
                sum_val += B[j, k] * C[k, i]
            C[j, i] = (A[j, i] - sum_val) / B[j, j]
    
    return B, C

def haltsky_solve(A, b):
    """Решение системы Ax = b методом Халецкого"""
    n = len(A)
    start_time = time.time()
    B, C = haltsky_decomposition(A)
    
    # Прямой ход: By = b
    y = np.zeros(n)
    for i in range(n):
        sum_val = 0.0
        for j in range(i):
            sum_val += B[i, j] * y[j]
        y[i] = (b[i] - sum_val) / B[i, i]
    
    # Обратный ход: Cx = y
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        sum_val = 0.0
        for j in range(i+1, n):
            sum_val += C[i, j] * x[j]
        x[i] = y[i] - sum_val
    
    execution_time = time.time() - start_time
    return x, B, C, execution_time

def verify_solution(A, b, x):
    """Проверка правильности решения"""
    Ax = A @ x
    residual = norm(Ax - b)
    relative_residual = residual / norm(b)
    return Ax, residual, relative_residual

def generate_test_matrix(n):
    """Генерация матрицы, удовлетворяющей условиям применимости метода Халецкого"""
    # Создаем нижнюю треугольную матрицу B с ненулевой диагональю
    B = np.zeros((n, n))
    for i in range(n):
        B[i, i] = i + 1
        for j in range(i):
            B[i, j] = np.random.uniform(-5, 5)
    
    # Создаем верхнюю треугольную матрицу C с единицами на диагонали
    C = np.eye(n)
    for i in range(n):
        for j in range(i+1, n):
            C[i, j] = np.random.uniform(-5, 5)
    
    # Формируем матрицу A = B * C
    A = B @ C
    b = np.random.uniform(-10, 10, n)
    return A, b

def main():
    # 💧 Проверка состояния приложения
    if 'started' not in st.session_state:
        st.session_state.started = False
    
    # 🌄 Стартовый экран с фоном из локального файла и liquid glass кнопкой
    if not st.session_state.started:
        # CSS стили для liquid glass эффекта
        st.markdown("""
        <style>
        .start-screen {
            text-align: center;
            padding: 2rem;
            height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        .app-title {
            font-size: 3rem;
            margin-bottom: 1rem;
            color: #1e3c72;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .app-subtitle {
            font-size: 1.5rem;
            color: #3a5a72;
            margin-bottom: 2rem;
        }
        .description {
            max-width: 600px;
            margin: 0 auto 2rem;
            line-height: 1.6;
            color: #4a6a82;
            background: rgba(255, 255, 255, 0.7);
            padding: 1.5rem;
            border-radius: 10px;
            backdrop-filter: blur(5px);
            -webkit-backdrop-filter: blur(5px);
        }
        .liquid-glass-btn {
            background: rgba(255, 255, 255, 0.25);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 16px;
            color: white !important;
            font-size: 1.2rem;
            font-weight: 600;
            padding: 12px 24px;
            margin: 20px auto;
            width: 220px;
            display: block;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .liquid-glass-btn:hover {
            background: linear-gradient(135deg, #5a6fd8 0%, #6a3d9a 100%);
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
        }
        .liquid-glass-btn:active {
            transform: translateY(0);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        </style>
        """, unsafe_allow_html=True)
        
        try:
            # 📁 Автоматическое определение пути к файлу
            import os
            import base64
            
            # Определяем путь к файлу (работает и локально, и в Streamlit Cloud)
            image_path = "background.png"
            
            # Проверяем, существует ли файл
            if os.path.exists(image_path):
                # Читаем файл и конвертируем в base64
                with open(image_path, "rb") as img_file:
                    bg_image_base64 = base64.b64encode(img_file.read()).decode()
                
                # Применяем фон через base64
                st.markdown(f"""
                <style>
                .start-screen {{
                    background: linear-gradient(rgba(255, 255, 255, 0.85), rgba(240, 248, 255, 0.85)), url("data:image/png;base64,{bg_image_base64}");
                    background-size: cover;
                    background-position: center;
                    background-repeat: no-repeat;
                }}
                </style>
                """, unsafe_allow_html=True)
            else:
                # Резервный градиентный фон, если файл не найден
                st.markdown("""
                <style>
                .start-screen {
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                }
                </style>
                """, unsafe_allow_html=True)
                st.warning("⚠️ Файл background.png не найден. Используется градиентный фон.")
        
        except Exception as e:
            # Резервный вариант при любой ошибке
            st.markdown("""
            <style>
            .start-screen {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            }
            </style>
            """, unsafe_allow_html=True)
            st.warning(f"⚠️ Ошибка загрузки фона: {str(e)}. Используется градиентный фон.")
        
        # Контент стартового экрана
        st.markdown("""
        <div class="start-screen">
            <h1 class="app-title">🧮 Добро пожаловать в решатель СЛАУ!</h1>
            <h2 class="app-subtitle">Метод Халецкого</h2>
            <div class="description">
                Это приложение решает системы линейных уравнений методом Халецкого.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Кнопка в стиле liquid glass (центрированная)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Начать работу", type="primary", use_container_width=True, key="start_btn"):
                st.session_state.started = True
                st.rerun()
        
        return  # Важно: останавливаем выполнение, пока пользователь не нажмёт "Начать"
    
    # 📊 Основной интерфейс БЕЗ ФОНА
    st.title("🧮 Схема Халецкого")
    st.markdown("### Решение систем линейных уравнений")
    
    mode = st.radio("Выберите режим", ["Ручной ввод", "Сгенерировать систему (n≥50)"], horizontal=True)
    
    if mode == "Сгенерировать систему (n≥50)":
        n = st.slider("Размер системы", min_value=50, max_value=100, value=50)
        
        if st.button("Сгенерировать и решить", type="primary"):
            with st.spinner("Генерация и решение системы..."):
                A, b = generate_test_matrix(n)
                start_time = time.time()
                x, B, C, exec_time = haltsky_solve(A, b)
                Ax, residual, rel_residual = verify_solution(A, b, x)
            
            st.success("✅ Система успешно решена!")
            st.markdown(f"**Время решения:** {exec_time:.6f} сек")
            st.markdown(f"**Относительная невязка:** {rel_residual:.2e}")
            
            # Вывод всей сгенерированной матрицы
            with st.expander("Показать сгенерированную матрицу A (все элементы)", expanded=False):
                st.markdown("#### Матрица коэффициентов A:")
                for i in range(n):
                    row_str = " ".join([f"{A[i,j]:.2f}" for j in range(n)])
                    st.text(f"Строка {i+1}: {row_str}")
            
            # Вывод всего вектора b
            with st.expander("Показать вектор правых частей b (все элементы)", expanded=False):
                st.markdown("#### Вектор правых частей b:")
                for i in range(n):
                    st.text(f"b[{i+1}] = {b[i]:.2f}")
            
            # Вывод всего решения
            with st.expander("Показать полное решение (все x)", expanded=True):
                st.markdown("#### Полное решение системы:")
                for i in range(n):
                    st.text(f"x[{i+1}] = {x[i]:.6f}")
            
            # Вывод проверки подстановкой для всех уравнений
            with st.expander("Показать проверку подстановкой (все уравнения)", expanded=False):
                st.markdown("#### Проверка подстановкой для всех уравнений:")
                for i in range(n):
                    st.text(f"Ур-е {i+1}: Ax = {Ax[i]:.6f}, b = {b[i]:.6f}, разница = {Ax[i]-b[i]:.2e}")
    
    else:  # Ручной ввод
        n = st.number_input("Размер системы", min_value=2, max_value=6, value=3)
        
        st.markdown("### Введите коэффициенты системы:")
        A = np.zeros((n, n))
        b = np.zeros(n)
        
        for i in range(n):
            cols = st.columns(n + 1)
            for j in range(n):
                A[i, j] = cols[j].number_input(f"a{i+1}{j+1}", value=0.0, key=f"a_{i}_{j}", step=1.0)
            b[i] = cols[n].number_input(f"b{i+1}", value=0.0, key=f"b_{i}", step=1.0)
        
        if st.button("Решить систему", type="primary"):
            try:
                # Проверка условий применимости
                try:
                    B_test, C_test = haltsky_decomposition(A)
                    st.success("✅ Условия применимости метода выполнены")
                except Exception as e:
                    st.error(f"❌ Ошибка: {str(e)}")
                    st.stop()
                
                # Решение системы
                x, B, C, exec_time = haltsky_solve(A, b)
                Ax, residual, rel_residual = verify_solution(A, b, x)
                
                # Вывод результатов
                st.markdown("### Результаты решения:")
                st.markdown(f"**Относительная невязка:** {rel_residual:.2e}")
                
                st.markdown("#### Вектор решения x:")
                for i in range(n):
                    st.markdown(f"x<sub>{i+1}</sub> = {x[i]:.6f}", unsafe_allow_html=True)
                
                st.markdown("#### Проверка подстановкой:")
                for i in range(n):
                    st.markdown(f"""
                    Уравнение {i+1}:  
                    ∑a<sub>{i+1}j</sub>x<sub>j</sub> = {Ax[i]:.6f},  b<sub>{i+1}</sub> = {b[i]:.6f},  
                    Разница = {Ax[i] - b[i]:.2e}
                    """, unsafe_allow_html=True)
                
                # Исследование скорости (для сравнения)
                if n >= 3:
                    st.markdown("### Скорость работы метода:")
                    start_time = time.time()
                    x_gauss = np.linalg.solve(A, b)
                    gauss_time = time.time() - start_time
                    halt_time = time.time() - start_time - gauss_time
                    
                    st.markdown(f"Метод Халецкого: {halt_time:.6f} сек")
                    st.markdown(f"Метод Гаусса (встроенная функция): {gauss_time:.6f} сек")
            
            except Exception as e:
                st.error(f"❌ Ошибка при решении: {str(e)}")

if __name__ == "__main__":
    main()









