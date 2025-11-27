import streamlit as st
import numpy as np
import time
from numpy.linalg import norm, eigvals

st.set_page_config(page_title="Сравнение методов", page_icon="🧮", layout="wide")

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
    return x, execution_time

def square_root_method(A, b):
    """Решение системы методом квадратных корней (Холецкого)"""
    n = len(A)
    start_time = time.time()
    
    # Проверка симметричности
    if not np.allclose(A, A.T, atol=1e-8):
        raise ValueError("Матрица не симметричная. Метод квадратных корней неприменим.")
    
    # Проверка положительной определенности через собственные значения
    eigenvalues = eigvals(A)
    min_eig = np.min(np.real(eigenvalues))
    if min_eig <= 1e-8:
        raise ValueError(f"Матрица не положительно определена (мин. собств. значение = {min_eig:.4e}).")
    
    # Разложение Холецкого: A = L * L^T
    L = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1):
            sum_val = sum(L[i, k] * L[j, k] for k in range(j))
            if i == j:
                L[i, j] = np.sqrt(A[i, i] - sum_val)
            else:
                L[i, j] = (A[i, j] - sum_val) / L[j, j]
    
    # Прямой ход: L * y = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - sum(L[i, j] * y[j] for j in range(i))) / L[i, i]
    
    # Обратный ход: L^T * x = y
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - sum(L[j, i] * x[j] for j in range(i+1, n))) / L[i, i]
    
    execution_time = time.time() - start_time
    return x, execution_time

def verify_solution(A, b, x):
    """Проверка правильности решения"""
    Ax = A @ x
    residual = norm(Ax - b)
    relative_residual = residual / norm(b)
    return Ax, residual, relative_residual

def generate_test_matrix(n):
    """Генерация симметричной положительно определенной матрицы для обоих методов"""
    # Создаем случайную матрицу
    M = np.random.randn(n, n)
    # Делаем матрицу симметричной и положительно определенной
    A = M.T @ M + n * np.eye(n)  # A = M^T * M + n*I
    
    # Генерируем вектор правых частей
    b = np.random.uniform(-10, 10, n)
    return A, b

def main():
    st.title("🧮 Сравнение методов решения СЛАУ")
    st.markdown("### Метод Халецкого vs Метод квадратных корней (Холецкого)")
    
    mode = st.radio("Выберите режим", ["Ручной ввод", "Сгенерировать систему (n≥50)"], horizontal=True)
    
    if mode == "Сгенерировать систему (n≥50)":
        n = st.slider("Размер системы", min_value=50, max_value=100, value=50)
        
        if st.button("Сгенерировать и решить", type="primary"):
            with st.spinner("Генерация и решение системы..."):
                A, b = generate_test_matrix(n)
                st.session_state.A = A
                st.session_state.b = b
                st.session_state.n = n
                
                # Решение методом Халецкого
                x_halt, time_halt = haltsky_solve(A, b)
                Ax_halt, res_halt, rel_res_halt = verify_solution(A, b, x_halt)
                
                # Решение методом квадратных корней
                x_sqroot, time_sqroot = square_root_method(A, b)
                Ax_sqroot, res_sqroot, rel_res_sqroot = verify_solution(A, b, x_sqroot)
                
                # Сохраняем результаты в session_state
                st.session_state.x_halt = x_halt
                st.session_state.x_sqroot = x_sqroot
                st.session_state.time_halt = time_halt
                st.session_state.time_sqroot = time_sqroot
                st.session_state.res_halt = rel_res_halt
                st.session_state.res_sqroot = rel_res_sqroot
                st.session_state.solved = True
            
            if st.session_state.solved:
                st.success("✅ Система успешно решена обоими методами!")
                
                # Сравнение времени выполнения
                st.markdown("### ⏱️ Сравнение времени выполнения")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Метод Халецкого", f"{st.session_state.time_halt:.6f} сек")
                with col2:
                    st.metric("Метод квадратных корней", f"{st.session_state.time_sqroot:.6f} сек")
                
                if st.session_state.time_halt < st.session_state.time_sqroot:
                    st.success(f"✅ Метод Халецкого быстрее в {st.session_state.time_sqroot/st.session_state.time_halt:.1f} раз!")
                else:
                    st.info(f"ℹ️ Метод квадратных корней быстрее в {st.session_state.time_halt/st.session_state.time_sqroot:.1f} раз")
                
                # Сравнение точности
                st.markdown("### 📏 Сравнение точности")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Относительная невязка (Халецкий)", f"{st.session_state.res_halt:.2e}")
                with col2:
                    st.metric("Относительная невязка (Квадратные корни)", f"{st.session_state.res_sqroot:.2e}")
                
                # Вывод полного решения
                with st.expander("Показать полное решение (все x)", expanded=True):
                    st.markdown("#### Решение методом Халецкого:")
                    for i in range(st.session_state.n):
                        st.text(f"x[{i+1}] = {st.session_state.x_halt[i]:.6f}")
                    
                    st.markdown("#### Решение методом квадратных корней:")
                    for i in range(st.session_state.n):
                        st.text(f"x[{i+1}] = {st.session_state.x_sqroot[i]:.6f}")
                
                # Сравнение решений
                st.markdown("### 📊 Сравнение решений")
                differences = st.session_state.x_halt - st.session_state.x_sqroot
                max_diff = np.max(np.abs(differences))
                st.markdown(f"**Максимальная разница между решениями:** {max_diff:.2e}")
                
                if max_diff < 1e-6:
                    st.success("✅ Решения практически совпадают!")
                else:
                    st.warning("⚠️ Решения различаются. Проверьте вычисления.")
    
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
                # Проверка условий применимости для Халецкого
                try:
                    B_test, C_test = haltsky_decomposition(A)
                    st.success("✅ Условия применимости для метода Халецкого выполнены")
                except Exception as e:
                    st.error(f"❌ Ошибка для метода Халецкого: {str(e)}")
                    st.stop()
                
                # Проверка условий для метода квадратных корней
                is_symmetric = np.allclose(A, A.T, atol=1e-8)
                eigenvalues = eigvals(A)
                min_eig = np.min(np.real(eigenvalues))
                is_pos_def = min_eig > 1e-8
                
                if is_symmetric and is_pos_def:
                    st.success(f"✅ Условия для метода квадратных корней выполнены (мин. собств. значение = {min_eig:.4e})")
                else:
                    if not is_symmetric:
                        st.warning("⚠️ Матрица не симметричная. Метод квадратных корней неприменим!")
                    if not is_pos_def:
                        st.warning(f"⚠️ Матрица не положительно определена (мин. собств. значение = {min_eig:.4e}).")
                
                # Решение методом Халецкого
                x_halt, time_halt = haltsky_solve(A, b)
                Ax_halt, res_halt, rel_res_halt = verify_solution(A, b, x_halt)
                
                # Решение методом квадратных корней (если применим)
                try:
                    x_sqroot, time_sqroot = square_root_method(A, b)
                    Ax_sqroot, res_sqroot, rel_res_sqroot = verify_solution(A, b, x_sqroot)
                    method_sqroot_applicable = True
                except Exception as e:
                    st.error(f"❌ Метод квадратных корней не применим: {str(e)}")
                    method_sqroot_applicable = False
                
                # Вывод результатов
                st.markdown("### 📋 Результаты решения")
                
                # Время выполнения
                st.markdown("#### Время выполнения:")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Халецкий", f"{time_halt:.6f} сек")
                with col2:
                    if method_sqroot_applicable:
                        st.metric("Квадратные корни", f"{time_sqroot:.6f} сек")
                    else:
                        st.metric("Квадратные корни", "Неприменим")
                
                # Точность
                st.markdown("#### Точность решения:")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Невязка (Халецкий)", f"{rel_res_halt:.2e}")
                with col2:
                    if method_sqroot_applicable:
                        st.metric("Невязка (Квадратные корни)", f"{rel_res_sqroot:.2e}")
                
                # Решение
                st.markdown("#### Вектор решения x:")
                for i in range(n):
                    result_text = f"x<sub>{i+1}</sub>:<br>Халецкий: {x_halt[i]:.6f}"
                    if method_sqroot_applicable:
                        result_text += f"<br>Квадратные корни: {x_sqroot[i]:.6f}"
                        result_text += f"<br>Разница: {abs(x_halt[i] - x_sqroot[i]):.2e}"
                    st.markdown(result_text, unsafe_allow_html=True)
                
                # Сравнение решений (если оба метода применимы)
                if method_sqroot_applicable:
                    st.markdown("### 📊 Сравнение решений")
                    differences = x_halt - x_sqroot
                    max_diff = np.max(np.abs(differences))
                    st.markdown(f"**Максимальная разница:** {max_diff:.2e}")
                    if max_diff < 1e-6:
                        st.success("✅ Решения практически совпадают!")
            
            except Exception as e:
                st.error(f"❌ Ошибка при решении: {str(e)}")
                st.exception(e)

if __name__ == "__main__":
    main()
