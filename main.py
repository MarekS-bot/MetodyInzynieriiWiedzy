import math
import numpy as np

from austria import *


if __name__ == "__main__":

    A = np.array([[1., 2., 0.], [2., 0., 2.], [2, 2, 0]])

    MIW.svd(A)

    # Q, R = MIW.QR_decomposition2(A)
    # print(f"Q: {Q}")
    # print(f"R: {R}")

    # vec1 = [1] * 8
    # vec2 = [1]*4 + [-1]*4
    # vec3 = [1]*2 + [-1]*2 + [0]*4
    # vec4 = [0]*4 + [1]*2 + [-1]*2
    # vec5 = [1, -1] + [0]*6
    # vec6 = [0, 0, 1, -1] + [0]*4
    # vec7 = [0]*4 + [1, -1] + [0]*2
    # vec8 = [0]*6 + [1, -1]
    # matrix = np.array([vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]).reshape(8, 8)
    #
    # print(MIW.dot_operator(vec1, vec1), end=" ")
    # print(MIW.dot_operator(vec2, vec2), end=" ")
    # print(MIW.dot_operator(vec3, vec3), end=" ")
    # print(MIW.dot_operator(vec4, vec4), end=" ")
    # print(MIW.dot_operator(vec5, vec5), end=" ")
    # print(MIW.dot_operator(vec6, vec6), end=" ")
    # print(MIW.dot_operator(vec7, vec7), end=" ")
    # print(MIW.dot_operator(vec8, vec8))
    #
    # print("Macierz: ")
    # print(matrix)
    # print("Macierz trans: ")
    # print(matrix.T)
    # print("Dot oper: ")
    # matrix_dot = MIW.dot_operator(matrix, matrix.T)
    # print(matrix_dot)
    #
    # inverse = np.linalg.inv(matrix_dot)
    # trans = inverse.T
    # xa = np.array([8, 6, 2, 3, 4, 6, 6, 5]).T
    # xb = np.dot(trans, xa)
    # print(xb)
    # xb = np.dot(inverse, xa)
    # print(xb)

    """
    sigma_i = sqrt( lambda_i )
    Av_1/sigma_1 = U_1
    (Av_1)T = v_1 T * AT
    (Avi)T(Avi) = v_i^T A^T * v_i * A = v_i^t * lambda_i * v_i
    A = U SIGMA VT
    MxN = MxM MxN NxN
    Lewe (środkowe) prawe
        
    AAT = [][] = (5-l)(8-l)-4 = l^2 -13l + 36
    d = 25
    l1 = 4
    l2 = 9
    sigma_1 = 2
    sigma_2 = 3
    
    AAT u1 - l1 u1 = 0
    (AAT - l1 I)u1 = 0
    [5 - 9   2  ] [u1^1]  
    [ 2    8 - 9] [u1^2] = 0
    -4ui^1 + 2u1^2 = 0
    2u1^1 - u1^2 = 0 / *2
    4u1^1 = 2u1^2
    2u1^1 = u1^2
    u1^1 = 1 => u1^2 = 2
    
    u1 = alpha[1 2]T
    aby u1 było 1
    u1 = 1/sqrt(5)[1 2]T
    
          [1 2]             [5 2 4]
          [2 0]   [1 2 0]   [2 4 0]
    ATA = [0 2] * [2 0 2] = [4 0 4]
    
    (ATA - lI)v = 0
    det(^^^) = (5-l)(4-l)(4-l) - 20(4-l) = (4-l)[(5-l)(4-l)-20]
    l = 4
    (5-l)(4-l)-20 = 0
    l^2 - 9l = 0
    
    l=9
    l=4
    l=0
    s1 = 3
    s2 = 2
    
    [1 2 0]
    [2 0 2]
    
    [5 - 9  2    4 ]    [v1^1]
    [2     4-9   0 ]    [v1^2]
    [4      0   4-9] =  [v1^3] = 0
    2v1^1 = 5v1^2
    4v1^1 = 5v1^3
    v1^1 = 5
    
    2*5 = 5v1^2
    v1^2 = 2
    v1^3 = 4
    
    v1 = alpha[5 2 4]T
    v1 = 1/3sqrt(5) [5 2 4]T
    v2 = 1/sqrt(5)[0 2 -1]T
    v3 = 1/3[-2 1 2]T
    A = U SIGMA VT
    A = 1/sqrt(5)[1 2][2 -1]
"""

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
    # print("\n\n")
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


