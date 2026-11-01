# -*- coding: utf-8 -*-
"""exercises_ch21.py --- решения на задачи 21.1--21.6 от Глава 21 на

    „Матрични методи в оптиката. Част I: Геометрична оптика“
    Александър Георгиев Гайдарджиев

Всяка задача пресмята числата от решението ѝ в книгата и ги сравнява с
отпечатаните стойности; при разминаване скриптът спира с AssertionError.
Ползва функциите от mathOptics.py (Листинги 21.1--21.3).

Стартиране:  python exercises_ch21.py      (около 3 s)
"""

import numpy as np

from mathOptics import S, L, system_matrix, check_unimodular, cardinal


def check(label, value, expected, tol):
    """Отпечатва стойността и проверява, че съвпада с тази от книгата."""
    ok = abs(value - expected) <= tol
    print("  %-34s %+.6g  (книгата: %+.6g)%s"
          % (label, value, expected, "" if ok else "   <-- РАЗМИНАВАНЕ"))
    assert ok, label


def secant(g, x0, x1, tol=1e-10, max_iter=50):
    """Секущ метод за g(x) = 0 --- същият като в Листинг 21.7
    (в mathOptics.py той е вътре в скрипта, затова е повторен тук)."""
    for _ in range(max_iter):
        g0, g1 = g(x0), g(x1)
        x2 = x1 - g1 * (x1 - x0) / (g1 - g0)
        if abs(x2 - x1) < tol:
            return x2
        x0, x1 = x1, x2
    raise RuntimeError("няма сходимост")


# Триплетът от таблица 21.1 и толерансите от Листинг 21.5
radii = [22.014, -435.760, -22.213, 20.292, 79.684, -18.395]
thick = [3.259, 6.008, 1.000, 4.750, 2.952]
index = [1.0, 1.6204, 1.0, 1.6200, 1.0, 1.6204, 1.0]
tol_r = [0.002 * abs(r) for r in radii]
tol_t = [0.02, 0.05, 0.02, 0.05, 0.02]
tol_n = [0.0005, 0.0005, 0.0005]


# =====================================================================
# Задача 21.1 --- плоскопаралелна пластинка
# =====================================================================

def plate(t, n):
    """Пластинка с дебелина t и показател n във въздух."""
    return S(t, n)


def task_1():
    print("Задача 21.1")
    M1 = plate(12.0, 1.5)
    M2 = system_matrix([np.inf, np.inf], [12.0], [1.0, 1.5, 1.0])
    assert np.allclose(M1, [[1, 8], [0, 1]])
    assert np.allclose(M1, M2)
    check("T = t/n, mm", M1[0, 1], 8.0, 1e-12)
    check("det", check_unimodular(M1), 1.0, 1e-12)
    check("изместване t - T, mm", 12.0 - M1[0, 1], 4.0, 1e-12)


# =====================================================================
# Задача 21.2 --- разстояние до образа и увеличение
# =====================================================================

def image(M, R):
    """Образ на предмет на разстояние R вляво от ОР_C1: (S, m)."""
    A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    S_ = -(A * R + B) / (C * R + D)
    return S_, A + S_ * C


def task_2():
    print("Задача 21.2")
    M = system_matrix([60.0, -40.0], [10.0], [1.0, 1.5, 1.0])
    s, m = image(M, 100.0)
    check("(б) S, mm", s, 92.15, 5e-3)
    check("(б) m", m, -0.911, 5e-4)
    s, m = image(system_matrix(radii, thick, index), 1000.0)
    check("(в) S, mm", s, 45.03, 5e-3)
    check("(в) m", m, -0.0520, 5e-5)


# =====================================================================
# Задача 21.3 --- „безразличната“ повърхност r2
# =====================================================================

def delta_f(radii_new):
    f0 = cardinal(system_matrix(radii, thick, index))["f"]
    return cardinal(system_matrix(radii_new, thick, index))["f"] - f0


def task_3():
    print("Задача 21.3")
    r = list(radii); r[1] += 10 * tol_r[1]        # 2 % вместо 0,2 %
    check("(а) толеранс на r2, mm", 10 * tol_r[1], 8.715, 5e-4)
    check("(а) Delta f, mm", delta_f(r), -0.051, 5e-4)
    r = list(radii); r[5] += tol_r[5] / 10        # 0,02 % вместо 0,2 %
    check("(б) толеранс на r6, mm", tol_r[5] / 10, 0.00368, 5e-6)
    check("(б) Delta f, mm", delta_f(r), -0.0143, 5e-5)


# =====================================================================
# Задача 21.4 --- Монте Карло за ЗФО
# =====================================================================

def task_4():
    print("Задача 21.4")
    c0 = cardinal(system_matrix(radii, thick, index))
    f0, b0 = c0["f"], c0["t2"]
    rng = np.random.default_rng(2026)             # както в Листинг 21.6
    N = 20_000
    f_v, b_v = np.empty(N), np.empty(N)
    for k in range(N):
        r = [x + rng.normal(0, s / 3) for x, s in zip(radii, tol_r)]
        t = [x + rng.normal(0, s / 3) for x, s in zip(thick, tol_t)]
        n = list(index)
        for j, s in enumerate(tol_n):
            n[2 * j + 1] += rng.normal(0, s / 3)
        c = cardinal(system_matrix(r, t, n))
        f_v[k], b_v[k] = c["f"], c["t2"]
    check("(а) средно ЗФО, mm", b_v.mean(), 42.432, 5e-4)
    check("(а) сигма ЗФО, mm", b_v.std(ddof=1), 0.0910, 5e-5)
    check("(б) RSS/3, mm", 0.2737 / 3, 0.0912, 5e-5)
    both = np.mean((abs(f_v - f0) <= 0.15) & (abs(b_v - b0) <= 0.15))
    check("(в) годни по f И по ЗФО, %", 100 * both, 87.7, 0.05)


# =====================================================================
# Задача 21.5 --- телеобектив
# =====================================================================

def tele(d):
    """Две тънки лещи: f_I = 100 mm, f_II = -40 mm, на разстояние d."""
    return L(-1 / 40) @ S(d) @ L(1 / 100)


def task_5():
    print("Задача 21.5")
    d = secant(lambda x: cardinal(tele(x))["f"] - 200.0, 70.0, 90.0)
    c = cardinal(tele(d))
    check("(б) d, mm", d, 80.0, 1e-6)
    check("(в) ЗФО, mm", c["t2"], 40.0, 1e-6)
    check("(в) дължина / f", (d + c["t2"]) / c["f"], 0.6, 1e-9)
    check("(в) s2, mm", c["s2"], -160.0, 1e-6)


# =====================================================================
# Задача 21.6 --- периодична система
# =====================================================================

def power_loop(M, N):
    P = np.eye(2)
    for _ in range(N):
        P = M @ P
    return P


def power_chebyshev(M, N):
    """M^N = u_N M - u_{N-1} I, u_{N+1} = s u_N - u_{N-1}."""
    s = M[0, 0] + M[1, 1]
    u_prev, u = 0.0, 1.0                          # u_0, u_1
    for _ in range(N - 1):
        u_prev, u = u, s * u - u_prev
    return u * M - u_prev * np.eye(2)


def three_ways(M, N=100):
    P1 = power_loop(M, N)
    P2 = np.linalg.matrix_power(M, N)
    P3 = power_chebyshev(M, N)
    rel = max(np.max(np.abs(P - P1) / np.abs(P1)) for P in (P2, P3))
    return P1, P2, P3, rel


def task_6():
    print("Задача 21.6")
    # (б) устойчива: s = 0,75
    M = S(50.0) @ L(1 / 40)
    P1, P2, P3, rel = three_ways(M)
    check("(б) s = A + D", M[0, 0] + M[1, 1], 0.75, 1e-12)
    for (i, j), v in zip([(0, 0), (0, 1), (1, 0), (1, 1)],
                         [1.193, -36.39, 0.01820, 0.2832]):
        check("(б) M^100[%d,%d]" % (i, j), P1[i, j], v, 5e-4 * abs(v))
    assert rel < 4e-14
    for P in (P1, P2, P3):
        check_unimodular(P, tol=5e-15)
    print("  (б) относителни разлики %.1e < 4e-14; det = 1 и за трите"
          % rel)

    # (в) неустойчива: s = -3
    M = S(50.0) @ L(1 / 10)
    P1, P2, P3, rel = three_ways(M)
    check("(в) s = A + D", M[0, 0] + M[1, 1], -3.0, 1e-12)
    lam = (3 + 5 ** 0.5) / 2
    check("(в) |lambda_1|^100 / 1e41", lam ** 100 / 1e41, 6.274, 5e-4)
    for (i, j), v in zip([(0, 0), (0, 1), (1, 0), (1, 1)],
                         [1.0151e42, -1.4029e43, 2.8057e40, -3.8774e41]):
        check("(в) M^100[%d,%d] / 1e40" % (i, j), P1[i, j] / 1e40,
              v / 1e40, 5e-5 * abs(v) / 1e40)
    assert rel < 1.4e-14
    print("  (в) относителни разлики %.1e < 1.4e-14" % rel)
    print("  (в) det: цикъл %.3g, matrix_power %.3g, Чебишов %.3g"
          % tuple(np.linalg.det(P) for P in (P1, P2, P3)))
    check("(в) det, matrix_power / 1e67",
          np.linalg.det(P2) / 1e67, 7.85, 5e-3)
    # препълване на двойната точност при цикъла
    P, N = np.eye(2), 0
    with np.errstate(over="ignore"):
        while np.all(np.isfinite(P)):
            P, N = M @ P, N + 1
    check("(в) първо N с препълване", N, 734, 0)


if __name__ == "__main__":
    for task in (task_1, task_2, task_3, task_4, task_5, task_6):
        task()
        print()
    print("всички стойности съвпадат с книгата")
