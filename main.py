import math
import numpy as np

from austria import *


if __name__ == "__main__":
    print("Hello")

# Main
# if __name__ == "__main__":
    # data = MIW()
    # data.show_data()
    #     # print(data.distance_index(0, 5))
    #     # print(data.distance_index(0, 15))
    #     # print(data.distance_index(0, 20))
    #     # x = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.0]
    #     # data.measure(x)
    #     # print(data.group[0.0][:5])  # 0 = 1216.27; 160.78; 280.50; 117.47
    #     # print(data.group[1.0][:5])  # 1 = 25.71; 170.50; 100.61; 564.69
    #     # print("\nF1:")
    #     # res_func1 = data.func1()
    #     # print(type(res_func1))
    #     # print_dl(res_func1)
    #     # print("\nF2:")
    #     # res_func2 = data.func2(res_func1, 5)
    #     # print_dl(res_func2)
    #     # print("\nF3:")
    #     # res_func3 = data.func3(res_func2)
    #     # print_dl(res_func3)
    #     # print("\nF4:")
    #     # res_func4 = data.func4(res_func3)
    #     # print(res_func4)
    #     # print()
    #     # print(MIW.metryka_euklidesowa_2(data.data[3], data.data[5]))
    #     # print(MIW.metryka_euklidesowa_2([1, 1], [2, 3]))
    #     # print(MIW.metryka_euklidesowa([1, 1], [2, 3]))
    #     #
    #     # HOMEWORK = METODA MONTECARLO DLA DOWOLNEJ FUNKCJI w przedziale
    #     # Metoda zaimplmenotwania prostokątów - oarametr jak kto chce [ilość podziałów lub dokładkość całokowania e]
    #     # Środek masy
    #
    #     """
    #         Odległość
    #         Środek masy = punkt, który ma najbliżej do innych kropki
    #         pokolorować na losowy kolor
    #         jeśli coś ma bliżej -> kolorowanie
    #         koniec, gdy już wszystko jest przekolorowane
    #         28 luty 1:10:00
    #     """
    #
    #     print()
    #
    #
    #     # def foo(x):
    #     #     return math.sin(x)
    #
    #
    #     # def egg(x):
    #     #     return math.cos(x)
    #     #
    #     #
    #     # def foo2(x):
    #     #     return x
    #
    #
    #     # print('Dla trojkata: ',
    #     #       round(
    #     #           MIW.monte_carlo(foo2, 1, 1, 100000), 2
    #     #       )
    #     #       )
    #
    #     # print('Dla sin(x) [monte carlo]: ', MIW.monte_carlo(foo, math.pi, 1, 1000))
    #     # print('Dla sin(x) [rectangles]: ', MIW.rectangles_values(foo, math.pi, 8))
    #     # print('Dla y=x [rectangles]: ', MIW.rectangles(foo2, 1, 4))
    #
    #     # zero, one = data.middle()
    #     # print(f'0: {zero}\n1: {one}')
    #     # if zero > one:
    #     #     print('% = ', round((one / (zero + one)) * 10, 2))
    #     # else:
    #     #     print('% = ', round((zero / (zero + one)) * 10, 2))
    #
    #     # vec = [1, 2, 3, 4, 5]
    #     # print(MIW.average(vec))
    #     # print(MIW.variation(vec))
    #
    #     # print('06.04.2022')
    #
    #     # def line(x):
    #     #     return (4 * x) + 3
    #
    #     """
    #     # (X)[jednostkowa + vector] * vector_beta[beta0, beta1]
    #     X/ X * B = Y
    #     B = 1/X * Y
    #     B = (XT X)^-1 XT Y
    #
    #     (2, 1)(5, 2)(7, 3)(8, 3)
    #     = 2/7, 5/14
    #     """
    #
    #     # print('\n\n')
    #     # x_matrix, y_matrix, matrix = MIW.find_betas_from_file()
    #     # print(f'X_Matrix: \n{x_matrix}')
    #     # print(f'Y_Matrix: \n{y_matrix.reshape((4,1))}')
    #     # print(f'Matrix: \n{matrix}')
    #
    #     # print(np.dot([1, 0], [0, 1]))
    #

    # matrix = np.array([[1, 1, 0], [0, 1, 1]])
    # print(matrix)
    # print(MIW.norma(matrix))

    """
    Wejsciowka:
    pisemny rozkład macierzy na dwie macierze

    proj_u(v) = (<u, v>/<u, u>) * u
    u1 = v1
    u2 = v2 - proj_u1(v2)
    e1 = u1/||u1||

    R = QT * A
    Q * R = Q * QT * A = (Q * QT)TT * A = (QT * Q)T * A = I^T A = A
    {QT * Q = I}
    """

    """
    Av = Lv
    Av - Lv = 0
    Av - LIv = 0
    (A-LI)v = 0
    
    [ a b ]      [ 1 0 ]   [ a - l    b   ]
    [ c d ]  - L [ 0 1 ] = [  c     d - l ]
    
    |   a - l     b     |
    |     c     d - l   | = (a - l)(d - l) - bc = ad - al - dl + l^2 - bc = l^2 - (a + d)l + ad - bc = 0
    
    Av = Lv
    (Av)^H = (Lv)^H
    v^H A^H = L^H v^H
    v^H A = L^H v^H / *v
    v^H A v = L^H v^H v
    L v^H v = L^H v^H v
    L = L^H
    
    A0 = A
    Ak = Qk Rk
    Ak+1 = Rk Qk
    kwadratowa
    
    Ak+1 = Rk Qk = I Rk Qk = (Q^T_k Qk) Rk Qk = Q^T_k Ak Qk = Q^-1+_k Ak Qk
    Wartości własne do obliczenia, żeby to zrobić korzystamy z rozkładu QR i dzięki temu wzorowi
    tworzymy przybliżenia wartości własnych
    [Macierz jest kwadratowa, rzeczywista, symetryczna]
    """

    # a=np.array([[1.,2.,3.,4.],[2.,2.,3.,4.],[3.,3.,3.,4.],[4.,4.,4.,4.]])
    # print(a)
    # wynik = MIW.matrix_eigenvalues(a)
    # print("================================")
    # print(np.round(wynik, decimals=3))

    # a=np.array([[1.,2.,3.,4.,5.],[2.,2.,3.,4.,5.],[3.,3.,3.,4.,5.],[4.,4.,4.,4.,5.],[5.,5.,5.,5.,5.]])
    # print(MIW.dekompozycja_Q(a))
    print("\n\n")
    # A = np.array([[6, 2], [3, 1]])
    # eigenvalues = MIW.matrix_eigenvalues(A)
    # eigenvalues = np.linalg.eig(A)
    # print(eigenvalues)
    # for value in eigenvalues:
    #     diagonal = np.diag(np.array([value] * A.shape[1]))
    #     A_sub = A - diagonal
    #     print(A_sub)

    # b_k = np.random.rand(A.shape[1])
    # for _ in range(1000):
    #     b_k1 = np.dot(A, b_k)
    #     b_k1_norm = np.linalg.norm(b_k1)
    #     b_k = b_k1 / b_k1_norm
    # print(b_k)

    """
    odcinek :=   30 -------- 50
    na:           0 -------- 1
    następnie:    0 -------- 10
    i koniec:    10 -------- 20

    Wysłać reozytoria
    Na następne:
    wykład 4 maja
    i tydzień wcześniej

    Odrabianie: następny tydzień
    """

    # A = np.array([[1, 1, 1, 9], [2, -3, 4, 13], [3, 4, 5, 40]])
    # x = MIW.gauss_elimination(A)
    # print(x)

    # A = np.array([[1., -1., 1., -1.], [1., 0., 0., 0.], [1., 1., 1., 1.], [1., 2., 4., 8.]])
    # B = np.array([[14.], [4.], [2.], [2.]])
    # print(MIW.gauss_elimination(np.copy(A), np.copy(B)))
