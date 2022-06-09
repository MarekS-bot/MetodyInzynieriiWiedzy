import typing as typ
import math
import numpy as np
import random


class MIW:

    # constructor
    def __init__(self):
        with open("australian.dat") as file:
            self.data = [list(map(lambda el: float(el), line.split())) for line in file]
        self.group: typ.Dict[float, typ.List[float]] = {}
        self.vectors: typ.List[typ.Dict[str, typ.List[float]]] = []
        self.random_arr = []

    # ===================================================================================== #

    def set_vectors(self):
        i = 1
        for el in self.data:
            self.vectors.append({f"p{i}": el})
            i += 1

    # ===================================================================================== #

    def show_data(self, head: int = 5) -> None:
        string: str = ""
        for i in range(head):
            string += f"Point nr{i}: {self.data[i]}\n"
        print(string)

    def show_group(self) -> str:
        string: str = ""
        for k, v in self.group.items():
            string += f"{k}: {v}\n"
        return string

    @staticmethod
    def round_down(n: float, decimals: int = 0):
        multiplier: int = 10**decimals
        return math.floor(n * multiplier) / multiplier

    # ===================================================================================== #

    @staticmethod
    def distance_between_points(p1: typ.List[float], p2: typ.List[float]) -> float:
        """Odległość pomiędzy dwoma punktami
        :param p1: punkt nr 1 : [wymiar_1, wymiar_2, wymiar_3 ... klasa decyzyjna]
        :param p2: punkt nr 2 : [wymiar_1, wymiar_2, wymiar_3 ... klasa decyzyjna]
        :return: float
        """
        res: float = 0.0
        for i in range(max(len(p1), len(p2)) - 1):
            res += pow(p1[i] - p2[i], 2)
        return math.sqrt(res)

    # ===================================================================================== #

    @staticmethod
    def euclidean_metric(p1, p2) -> float:
        res = 0.0
        for i in range(len(p1)):
            res += pow(p2[i] - p1[i], 2)
        res = math.sqrt(res)
        return res

    @staticmethod
    def euclidean_metric_dot_oper(vec1, vec2, dec=False):
        if dec:
            v1 = np.array(vec1[:-1])
            v2 = np.array(vec2[:-1])
        else:
            v1 = np.array(vec1)
            v2 = np.array(vec2)
        a = v2 - v1
        res = np.sqrt(np.dot(a, a))
        return res

    # ===================================================================================== #

    def _distance_index(self, p1: int, p2: int) -> float:
        res: float = 0.0
        for i in range(max(len(self.data[p1]), len(self.data[p2])) - 1):
            res += pow(self.data[p1][i] - self.data[p2][i], 2)
        return MIW.round_down(math.sqrt(res), 2)

    def grouping(self, decision_col: int) -> None:
        y_point: int = 0
        for x_point in range(1, len(self.data)):
            decision = self.data[x_point][decision_col]
            if decision in self.group.keys():
                self.group[decision].append(self._distance_index(y_point, x_point))
            else:
                self.group[decision] = [self._distance_index(y_point, x_point)]

    # ===================================================================================== #

    def measure(self, x_point: list):
        """
        Mierzy odległość między x_point, a innymi punktami w macierzy
        :param x_point: z poza, czy coś [wymiar_1, wymiar_2, wymiar_3 ... klasa_decyzyjna]
        :return: groupa <klucz: klasa decyzyjna: odległości>
        """
        for i in range(len(self.data)):
            choice: float = self.data[i][-1]
            if choice in self.group.keys():
                self.group[choice].append(
                    MIW.distance_between_points(x_point, self.data[i])
                )
            else:
                self.group[choice] = [
                    MIW.distance_between_points(x_point, self.data[i])
                ]

    def group_to_list_of_tuples(self) -> typ.List[typ.Tuple[float, typ.List[float]]]:
        """
        func1
         :return: Lista, gdzie jej elementy to tuple()
                pierwszy element tuple: to klasa decyzyjna, druga: lista odległości
        """
        res: typ.List[typ.Tuple[float, typ.List[float]]] = []
        for k, v in self.group.items():
            res.append((k, v))
        return res

    def list_of_tuples_to_dictionary_of_list(
        self, lis: typ.List[typ.Tuple[float, typ.List[float]]], k: int
    ) -> typ.Dict[float, typ.List[float]]:
        """
        func2
        Zamienia List[Tuple] => Dictionary <klucz: klasa decyzyjna. elementy: najmniejsze odległości <ilość: k>
        :param lis: to ta lista(tuple)
        :param k: ile chcemy odległości
        :return: zwraca dictionary{klasa_decyzyjna: lista[najmniejszych odległości]}
        """
        dicto_c_list: typ.Dict[float, typ.List[float]] = {}
        for elem in lis:
            c, v = elem
            v.sort()
            dicto_c_list[c] = v[:k]
        return dicto_c_list

    def dic_of_list_to_dic_of_kv(
        self, dicto: typ.Dict[float, list[float]]
    ) -> typ.Dict[float, float]:
        """
        func3
        Zamienia tą dictionary na dictionary, gdzie kluczem jest: klasa_decyzyjna, a elementami jest suma tych odległości
        :param dicto: ten słownik z poprzedniej funkcji
        :return: dicto
        """
        dicto_c_odl: typ.Dict[float, float] = {}
        for k, v in dicto.items():
            dicto_c_odl[k] = sum(v)
        return dicto_c_odl

    def dic_to_number_or_none(
        self, dicto: typ.Dict[float, float]
    ) -> typ.Union[int, None]:
        """func4"""
        keys: typ.List[float] = list(dicto.keys())
        if dicto[keys[0]] == dicto[keys[1]]:
            return None
        else:
            return 0

    # ===================================================================================== #

    @staticmethod
    def _middle_set_dis(values):
        """Function for the middle function"""
        result = {}
        for i in range(len(values)):
            point = values[i]
            point_key = list(point.keys())[0]
            point_val = list(point.values())[0]
            for j in range(len(values)):
                new_point = values[j]
                if new_point != point:
                    new_point_key = list(new_point.keys())[0]
                    new_point_val = list(new_point.values())[0]
                    if point_key in result.keys():
                        result[point_key].append(
                            {
                                new_point_key: MIW.euclidean_metric_dot_oper(
                                    point_val, new_point_val, True
                                )
                            }
                        )
                    else:
                        result[point_key] = [
                            {
                                new_point_key: MIW.euclidean_metric_dot_oper(
                                    point_val, new_point_val, True
                                )
                            }
                        ]
        return result

    @staticmethod
    def _mid_set_min(values):
        """Function for the middle function"""
        min_g = {}
        for point_from in values:
            for point_to in values[point_from]:
                point_to_val = list(point_to.values())[0]
                if point_from in min_g.keys():
                    min_g[point_from] += point_to_val
                else:
                    min_g[point_from] = point_to_val
        return min(min_g, key=min_g.get)

    def _get_pointer(self, vec, vector_arr):
        """Get pointer from vector_array"""
        for point in vector_arr:
            point_key = list(point.keys())[0]
            point_val = list(point.values())[0]
            if point_key == vec:
                return {point_key: point_val}

    def _set_random(self):
        """Setting the random array"""
        i = 1
        res = [entry[:14] + [float(random.choice([0, 1]))] for entry in self.data]
        for el in res:
            self.random_arr.append({f"p{i}": el})
            i += 1

    def middle(self):
        if not self.random_arr:
            self._set_random()

        number_zero = 0
        number_one = 0
        made_changes = 0

        iter = 0
        while True:
            iter += 1
            print(iter, end=" ")
            color = {"0": [], "1": []}

            for vec in self.random_arr:
                key = list(vec.keys())[0]
                val = list(vec.values())[0]
                decision = val[-1]
                if decision == 0.0:
                    color["0"].append({key: val})
                elif decision == 1.0:
                    color["1"].append({key: val})

            zeros = color["0"]
            ones = color["1"]

            dis_zero = MIW._middle_set_dis(zeros)
            dis_one = MIW._middle_set_dis(ones)

            min_zero = MIW._mid_set_min(dis_zero)
            min_one = MIW._mid_set_min(dis_one)

            min_zero_pointer = self._get_pointer(min_zero, self.random_arr)
            min_one_pointer = self._get_pointer(min_one, self.random_arr)

            closest = {
                "0": [],
                "1": [],
            }

            for point in self.random_arr:
                point_key = list(point.keys())[0]
                point_val = list(point.values())[0]
                if point_key != min_zero or point_key != min_one:
                    m_zero = MIW.euclidean_metric_dot_oper(
                        min_zero_pointer[min_zero], point_val, True
                    )
                    m_one = MIW.euclidean_metric_dot_oper(
                        min_one_pointer[min_one], point_val, True
                    )
                    if m_zero < m_one:
                        if not point_val[-1] == 0.0:
                            point_val[-1] = 0.0
                            made_changes += 1
                        closest["0"].append({point_key: point_val})
                    else:
                        if not point_val[-1] == 1.0:
                            point_val[-1] = 1.0
                            made_changes += 1
                        closest["1"].append({point_key: point_val})

            if made_changes == 0:
                print("\nResult: ")
                break
            made_changes = 0

            number_zero = len(closest["0"])
            number_one = len(closest["1"])

        return number_zero, number_one

    # ===================================================================================== #

    @staticmethod
    def cross_product(vec1, vec2):
        dim = len(vec1)
        res = []
        for i in range(dim):
            if i == 0:
                j, k = 1, 2
                res.append(vec1[j] + vec2[k] - vec1[k] + vec2[j])
            elif i == 1:
                j, k = 2, 0
                res.append(vec1[j] + vec2[k] - vec1[k] + vec2[j])
            else:
                j, k = 0, 1
                res.append(vec1[j] + vec2[k] - vec1[k] + vec2[j])
        return res

    @staticmethod
    def dot_operator(vec1, vec2):
        return np.dot(vec1, vec2)

    @staticmethod
    def average(vector: typ.List[float]) -> float:
        return np.dot(vector, np.ones(len(vector))) / len(vector)

    @staticmethod
    def variation(vector: typ.List[float]) -> float:
        average = MIW.average(vector)
        avg_metric = np.ones(len(vector)) * average
        diff = vector - avg_metric
        return (np.dot(diff, diff)) / len(vector)

    # ===================================================================================== #

    @staticmethod
    def average_2(vec):
        sum = 0.0
        for elem in vec:
            sum += elem
        return float(sum / float(len(vec)))

    @staticmethod
    def variation_2(vector):
        average = MIW.average_2(vector)
        sum = 0.0
        for elem in vector:
            sum += (elem - average) ** 2
        return float(sum / float(len(vector)))

    @staticmethod
    def standard(vec):
        return np.sqrt(MIW.variation_2(vec))

   # ===================================================================================== #

    @staticmethod
    def monte_carlo(foo, limit_x, limit_y, count):
        point_in = 0
        for c in range(count):
            x = random.uniform(0, limit_x)
            y = random.uniform(0, limit_y)
            value_x = abs(foo(x))
            if 0 < y <= value_x:
                point_in += 1
        return (point_in / count) * (limit_x * limit_y)

    @staticmethod
    def rectangles_values(foo, limit_x, count):
        dx = limit_x / count
        rects = []
        for i in range(1, count + 1):
            if dx * i > limit_x:
                x_k = limit_x
            else:
                x_k = dx * i
            y_k = abs(foo(x_k))
            rects.append(y_k)
        return dx * sum(rects)

    @staticmethod
    def rectangles_area(foo, limit_x, count):
        dx = limit_x / count
        area = []
        for i in range(1, count + 1):
            if dx * i > limit_x:
                x_k = limit_x
            else:
                x_k = dx * i
            y_k = foo(x_k)
            x_p = dx * (i - 1)
            y_p = abs(foo(x_p))
            limit_y = max([y_p, y_k])
            area.append((x_k - x_p) * limit_y)
        return dx * sum(area)

    @staticmethod
    def rectangles(foo, limix_x, count):
        return random.uniform(
            MIW.rectangles_values(foo, limix_x, count),
            MIW.rectangles_area(foo, limix_x, count),
        )

    # ===================================================================================== #

    @staticmethod
    def norma(vec1, vec2):
        return np.sqrt(MIW.dot_operator(vec1, vec2))

    @staticmethod
    def find_betas_from_file():
        x = []
        y = []
        with open("vectors.txt") as file:
            for line in file:
                line = line.replace("(", "")
                line = line.replace(")", "")
                tup = tuple(map(int, line.split(", ")))
                x.append(tup[0])
                y.append(tup[1])

        y_matrix = np.array(y)
        x_matrix = np.array([[1, el] for el in x])

        matrix = np.dot(x_matrix.T, x_matrix)
        matrix = np.linalg.inv(matrix)
        matrix = np.dot(matrix, x_matrix.T)
        matrix = np.dot(matrix, y_matrix)
        return x_matrix, y_matrix, matrix

    # ===================================================================================== #

    @staticmethod
    def proj(u, v):
        u_v = np.dot(u.T, v)
        u_u = np.dot(u.T, u)
        if u_u == 0:
            return u
        return (u_v / u_u) * u


# ===================================================================================== #

    @staticmethod
    def Q_decomposition(A: np.ndarray):
        n, m = A.shape
        u = np.zeros((n, m))
        u[:, 0] = A[:, 0] / np.linalg.norm(A[:, 0])
        Q = np.zeros((n, m))
        Q[:, 0] = u[:, 0]
        for i in range(1, m):
            u_prev = u[:, i - 1]
            u[:, i] = A[:, i] - (
                    np.dot(A[:, i].T, u_prev) / np.dot(u_prev.T, u_prev)
            ) * u_prev
            Q[:, i] = u[:, i] / np.linalg.norm(u[:, i])
        return Q

    @staticmethod
    def QR_decomposition(A: np.ndarray):
        Q = MIW.Q_decomposition(A)
        R = np.dot(Q.T, A)
        return Q, R

# ===================================================================================== #

    @staticmethod
    def gauss_elimination(A, B):
        n = len(A)
        for i in range(n-1):
            for j in range(i+1, n):
                ratio = A[j][i]/A[i][i]
                A[j][i] = ratio
                for k in range(i+1, n):
                    A[j][k] = A[j][k] - ratio * A[i][k]
                B[j] = B[j] - ratio * B[i]
        x = np.zeros(n)
        k = n-1
        x[k] = B[k]/A[k, k]
        while k >= 0:
            x[k] = (B[k] - np.dot(A[k, k+1:], x[k+1:]))/A[k, k]
            k -= 1
        return x

# ===================================================================================== #

    # @staticmethod
    # def matrix_eigenvalues(a):
    #     new_a = a
    #     i = 0
    #     while (np.diag(new_a) - np.dot(new_a, np.ones((new_a.shape[0], 1))).T).all() > 0.001:
    #         q = MIW.Q_decomposition(new_a)
    #         new_a = np.dot(np.dot(q.T, a), q)
    #         i += 1
    #     return np.diag(new_a)


    @staticmethod
    def svd(A: np.ndarray):
        n, m = A.shape
        U = np.zeros((n, n))
        S = np.zeros((n, m))
        V = np.zeros((m, m))

        eigen_val, eigen_vec = np.linalg.eig(np.dot(A, A.T)) # [4., 9.]
        for i in range(len(eigen_val)):
            S[i, i] = np.sqrt(eigen_val[i])
        for i in range(len(eigen_vec)):
            U[:, i] = 1/np.linalg.norm(eigen_vec[i]) * eigen_vec[i]
        S_inv = np.zeros((n, m))
        for i in range(S.shape[0]):
            S_inv[i, i] = 1/S[i, i]
        V = np.dot(np.dot(A.T, U), S_inv)
        VT = V.T
        return U, S, VT
