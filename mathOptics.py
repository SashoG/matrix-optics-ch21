# -*- coding: utf-8 -*-
"""mathOptics.py --- придружаващ код към Глава 21 на

    „Матрични методи в оптиката. Част I: Геометрична оптика“
    Александър Георгиев Гайдарджиев

Файлът съдържа ТОЧНО кода от листингите на главата плюс кратък скрипт,
който възпроизвежда числата от таблици 21.2--21.3, фиг. 21.3,
Листинг 21.7 и останалите числа от §21.7 (бисекцията, стъпките на
секущия метод и компенсатора). Задачите 21.1--21.6 са във файла
exercises_ch21.py.

Изисква Python >= 3.6 и numpy >= 1.17 (заради np.random.default_rng);
методът на Брент изисква и scipy (без него разделът се пропуска).

Стартиране:  python mathOptics.py          (около 10 s)
Тестовете:   python -m pytest mathOptics.py

Конвенции (както в книгата):
    лъчев вектор   (y, V),  V = n * theta
    преместване    S(T) = [[1, T], [0, 1]],  T = t / n   (приведена дебелина)
    пречупване     R(P) = [[1, 0], [-P, 1]], P = (n' - n) / r
    матрица на оптичната система: последният елемент стои НАЙ-ОТЛЯВО.
Всички дължини са в милиметри.
"""

import numpy as np


# =====================================================================
# Листинг 21.1 --- базовите матрици
# =====================================================================

def S(t, n=1.0):
    """Матрица на преместване с приведена дебелина T = t/n."""
    return np.array([[1.0, t / n], [0.0, 1.0]])


def R(r, n1, n2):
    """Матрица на пречупване от сферична повърхност с радиус r."""
    P = (n2 - n1) / r              # оптична сила на повърхността
    return np.array([[1.0, 0.0], [-P, 1.0]])


def L(P):
    """Матрица на тънка леща с оптична сила P = 1/f."""
    return np.array([[1.0, 0.0], [-P, 1.0]])


# =====================================================================
# Листинг 21.2 --- сглобяване на матрицата на оптичната система
# =====================================================================

def system_matrix(radii, thicknesses, indices):
    """Собствена матрица на центрирана система.

    radii       -- r_1 ... r_N (радиуси на повърхностите)
    thicknesses -- t_1 ... t_{N-1} (разстояния МЕЖДУ повърхностите)
    indices     -- n_0 ... n_N (среди: n_0 отляво, n_N отдясно)
    """
    M = np.eye(2)
    for i, r in enumerate(radii):
        M = R(r, indices[i], indices[i + 1]) @ M        # пречупване
        if i < len(thicknesses):
            M = S(thicknesses[i], indices[i + 1]) @ M   # преместване
    return M


def check_unimodular(M, tol=1e-9):
    """Вграден тест: детерминантата на всяка матрица трябва да е 1."""
    d = np.linalg.det(M)
    if abs(d - 1.0) > tol:
        raise ValueError("det M = %.12f, а трябва да е 1" % d)
    return d


# =====================================================================
# Листинг 21.3 --- кардинални елементи (формулите от глава 9)
# =====================================================================

def cardinal(M, n1=1.0, n2=1.0):
    """Кардинални елементи на система със собствена матрица M."""
    A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    if abs(C) < 1e-12:
        raise ValueError("афокална система: C = 0, няма фокуси")
    return {
        "P":  -C,                   # оптична сила
        "f":  -n2 / C,              # фокусно разстояние
        "t1":  n1 * D / C,          # ОР_C1 -> преден фокус
        "t2": -n2 * A / C,          # ОР_C2 -> заден фокус (ЗФО)
        "s1":  n1 * (D - 1.0) / C,  # ОР_C1 -> предна главна точка
        "s2":  n2 * (1.0 - A) / C,  # ОР_C2 -> задна главна точка
    }


# =====================================================================
# Листинг 21.8 --- автоматични тестове
# =====================================================================

def test_thin_lens():
    """Тънка леща с f = 50 mm: A = D = 1, B = 0, C = -1/f."""
    M = L(1 / 50.0)
    assert np.allclose(M, [[1, 0], [-0.02, 1]])
    assert abs(cardinal(M)["f"] - 50.0) < 1e-12


def test_thick_lens_example():
    """Дебелата леща от Пример 9.1: f = 49,66 mm, ЗФО = +46,90 mm."""
    M = system_matrix([60.0, -40.0], [10.0], [1.0, 1.5, 1.0])
    check_unimodular(M)
    c = cardinal(M)
    assert abs(c["f"] - 49.655) < 5e-3
    assert abs(c["t2"] - 46.897) < 5e-3


def test_afocal_raises():
    """Кеплеров телескоп: C = 0, кардинални точки не съществуват."""
    M = L(1 / 25.0) @ S(175.0) @ L(1 / 150.0)
    assert abs(M[1, 0]) < 1e-12
    try:
        cardinal(M)
    except ValueError:
        pass
    else:
        raise AssertionError("афокалната система трябва да даде грешка")


# =====================================================================
# Скрипт: възпроизвежда числата от главата
# =====================================================================

if __name__ == "__main__":
    test_thin_lens()
    test_thick_lens_example()
    test_afocal_raises()
    print("тестовете минаха\n")

    # --- Листинг 21.4: триплетът от таблица 21.1 ---------------------
    # Конструктивни данни на триплета (таблица 21.1), всичко в милиметри
    radii = [22.014, -435.760, -22.213, 20.292, 79.684, -18.395]
    thick = [3.259, 6.008, 1.000, 4.750, 2.952]
    index = [1.0, 1.6204, 1.0, 1.6200, 1.0, 1.6204, 1.0]

    M = system_matrix(radii, thick, index)
    check_unimodular(M)
    c = cardinal(M)
    print("--- Таблица 21.2: кардинални елементи ---")
    print("P   = %.6f 1/mm" % c["P"])
    print("f   = %+.3f mm" % c["f"])
    print("ЗФО = %+.3f mm" % c["t2"])
    print("ПФО = %+.3f mm" % c["t1"])
    print("s1  = %+.3f mm" % c["s1"])
    print("s2  = %+.3f mm" % c["s2"])

    # --- Листинг 21.5: чувствителност --------------------------------
    tol_r = [0.002 * abs(r) for r in radii]   # радиуси: 0,2 %
    tol_t = [0.02, 0.05, 0.02, 0.05, 0.02]    # стъкло 0,02; въздух 0,05 mm
    tol_n = [0.0005, 0.0005, 0.0005]          # показатели на пречупване

    def f_and_bfd(radii, thick, index):
        c = cardinal(system_matrix(radii, thick, index))
        return c["f"], c["t2"]

    f0, b0 = f_and_bfd(radii, thick, index)
    rows = []
    for i, dr in enumerate(tol_r):            # шестте радиуса
        r = list(radii); r[i] += dr
        f, b = f_and_bfd(r, thick, index)
        rows.append(("r%d" % (i + 1), dr, f - f0, b - b0))
    for i, dt in enumerate(tol_t):            # петте разстояния
        t = list(thick); t[i] += dt
        f, b = f_and_bfd(radii, t, index)
        rows.append(("t%d" % (i + 1), dt, f - f0, b - b0))
    for k, dn in enumerate(tol_n):            # трите стъкла
        n = list(index); n[2 * k + 1] += dn
        f, b = f_and_bfd(radii, thick, n)
        rows.append(("n%d" % (k + 1), dn, f - f0, b - b0))

    print("\n--- Таблица 21.3: чувствителност ---")
    for name, tol, df, db in rows:
        print("%-3s %8.4f %+9.4f %+9.4f" % (name, tol, df, db))
    print("най-лош случай: %.4f  %.4f"
          % (sum(abs(x[2]) for x in rows), sum(abs(x[3]) for x in rows)))
    rss_f = sum(x[2]**2 for x in rows)**0.5
    rss_b = sum(x[3]**2 for x in rows)**0.5
    print("RSS:            %.4f  %.4f" % (rss_f, rss_b))

    # --- Листинг 21.6: Монте Карло -----------------------------------
    rng = np.random.default_rng(2026)   # фиксиран seed -> повторим опит
    N = 20_000
    f_values = np.empty(N)
    for k in range(N):
        r = [x + rng.normal(0, s / 3) for x, s in zip(radii, tol_r)]
        t = [x + rng.normal(0, s / 3) for x, s in zip(thick, tol_t)]
        n = list(index)
        for j, s in enumerate(tol_n):
            n[2 * j + 1] += rng.normal(0, s / 3)
        f_values[k] = cardinal(system_matrix(r, t, n))["f"]

    print("\n--- Фигура 21.3: Монте Карло ---")
    print("средно %.4f mm" % f_values.mean())
    print("сигма  %.4f mm" % f_values.std(ddof=1))
    dopusk = 0.15
    godni = np.mean(np.abs(f_values - f0) <= dopusk)
    print("годни при +-%.2f mm: %.1f %%" % (dopusk, 100 * godni))

    # --- Листинг 21.7: оптимизация (секущ метод) ---------------------
    def f_of_t4(t4):
        t = list(thick); t[3] = t4
        return cardinal(system_matrix(radii, t, index))["f"]

    def secant(g, x0, x1, tol=1e-10, max_iter=50):
        """Секущ метод за решаване на g(x) = 0."""
        for _ in range(max_iter):
            g0, g1 = g(x0), g(x1)
            x2 = x1 - g1 * (x1 - x0) / (g1 - g0)
            if abs(x2 - x1) < tol:
                return x2
            x0, x1 = x1, x2
        raise RuntimeError("няма сходимост")

    t4 = secant(lambda x: f_of_t4(x) - 50.0, 4.0, 5.5)
    print("\n--- §21.7: оптимизация (секущ метод) ---")
    print("t4 = %.6f mm  (промяна %+.4f mm)" % (t4, t4 - thick[3]))
    print("проверка: f = %.9f mm" % f_of_t4(t4))

    try:
        from scipy.optimize import brentq
        print("brentq:  t4 = %.6f mm"
              % brentq(lambda x: f_of_t4(x) - 50.0, 3.0, 7.0, xtol=1e-12))
    except ImportError:
        print("(scipy не е инсталиран --- методът на Брент е пропуснат)")

    # =================================================================
    # §21.7 (продължение) --- извън листингите
    # =================================================================

    # --- секущ метод стъпка по стъпка: пълна точност за пет стъпки ---
    x0, x1 = 4.0, 5.5
    for k in range(1, 6):
        g0, g1 = f_of_t4(x0) - 50.0, f_of_t4(x1) - 50.0
        x0, x1 = x1, x1 - g1 * (x1 - x0) / (g1 - g0)
        print("секущ, стъпка %d: t4 = %.12f mm, g = %+.1e mm"
              % (k, x1, f_of_t4(x1) - 50.0))

    # --- бисекция в [3; 7] mm, осем стъпки (за сравнение) ---
    a, b = 3.0, 7.0
    for _ in range(8):
        m = (a + b) / 2
        if (f_of_t4(a) - 50.0) * (f_of_t4(m) - 50.0) <= 0:
            b = m
        else:
            a = m
    print("бисекция, 8 стъпки: t4 = %.6f mm, f = %.4f mm" % (m, f_of_t4(m)))

    # --- компенсатор: всеки екземпляр се донастройва по t4 до f0 ---
    # Генераторът се пуска наново със същото начално число и тегли в
    # същия ред като Листинг 21.6, така че екземплярите са СЪЩИТЕ.
    def f_at(r, t, n, t4):
        t = list(t); t[3] = t4
        return cardinal(system_matrix(r, t, n))["f"]

    rng = np.random.default_rng(2026)
    dt4 = np.empty(N)
    bfd_before = np.empty(N)
    bfd_after = np.empty(N)
    for k in range(N):
        r = [x + rng.normal(0, s / 3) for x, s in zip(radii, tol_r)]
        t = [x + rng.normal(0, s / 3) for x, s in zip(thick, tol_t)]
        n = list(index)
        for j, s in enumerate(tol_n):
            n[2 * j + 1] += rng.normal(0, s / 3)
        c = cardinal(system_matrix(r, t, n))
        assert c["f"] == f_values[k]          # същият екземпляр
        bfd_before[k] = c["t2"]
        t4k = secant(lambda x: f_at(r, t, n, x) - f0, t[3], t[3] + 0.01)
        dt4[k] = t4k - t[3]                   # нужната поправка
        t[3] = t4k
        bfd_after[k] = cardinal(system_matrix(r, t, n))["t2"]

    print("\n--- §21.7: компенсатор (донастройка до f0 = %.3f mm) ---" % f0)
    print("поправка dt4 от %+.3f до %+.3f mm, сигма %.3f mm"
          % (dt4.min(), dt4.max(), dt4.std(ddof=1)))
    print("ЗФО, сигма:  %.3f -> %.3f mm"
          % (bfd_before.std(ddof=1), bfd_after.std(ddof=1)))
    print("ЗФО, годни при +-%.2f mm: %.1f %% -> %.1f %%"
          % (dopusk, 100 * np.mean(np.abs(bfd_before - b0) <= dopusk),
             100 * np.mean(np.abs(bfd_after - b0) <= dopusk)))
