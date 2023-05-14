import dill
import numpy as np

rot = np.array([[+0, +1], [-1, +0]])

G1_hat = np.array(
    [
        [-np.sqrt(1 / 2), -np.sqrt(1 / 2)],
        [-np.sqrt(1 / 2), +np.sqrt(1 / 2)],
    ]
)

G2_hat = np.array(
    [
        [-1, +0],
        [+0, +1],
    ]
)

G3_hat = np.array([[+0, +1], [+1, +0]])


def orient(arg):
    indices = np.argsort(arg[:3])
    oriented = np.concatenate([arg[:3][indices], arg[3:][indices]])

    return oriented


def get_middle_indices(num_points, triangles):
    is_middle = np.zeros(num_points, dtype=np.bool_)
    for idx, elem in enumerate(triangles):
        is_middle[elem[3:]] = True

    return is_middle


def combine_arguments(points, element, right_part_values):
    local_triangle = points[element]

    # ----------------P's----------------------------------------------------------------
    x1, x2, x3 = points[element[0], 0], points[element[1], 0], points[element[2], 0]
    y1, y2, y3 = points[element[0], 1], points[element[1], 1], points[element[2], 1]

    delta = x1 * y2 - x1 * y3 - x2 * y1 + x2 * y3 + x3 * y1 - x3 * y2

    P_1_x = (-y1 + y3) / delta
    P_1_y = (+x1 - x3) / delta

    P_2_x = (+y1 - y2) / delta
    P_2_y = (-x1 + x2) / delta
    # ------------------------------------------------------------------------------------

    # ----------------------Lengths-------------------------------------------------------
    l1 = ((x2 - x3) ** 2 + (y2 - y3) ** 2) ** 0.5
    l2 = ((x1 - x3) ** 2 + (y1 - y3) ** 2) ** 0.5
    l3 = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    # ------------------------------------------------------------------------------------

    # ----------------------Jacobian------------------------------------------------------
    J = np.array([[P_1_x, P_1_y], [P_2_x, P_2_y]])
    J_inv_T = np.linalg.inv(J).T
    # -----------------------------------------------------------------------------------

    # -----------------------nomal-and-tangential-vectors--------------------------------
    t1 = local_triangle[2] - local_triangle[1]
    t1 = t1 / np.linalg.norm(t1)
    n1 = rot @ t1

    t2 = local_triangle[2] - local_triangle[0]
    t2 = t2 / np.linalg.norm(t2)
    n2 = rot @ t2

    t3 = local_triangle[1] - local_triangle[0]
    t3 = t3 / np.linalg.norm(t3)
    n3 = rot @ t3
    # ------------------------------------------------------------------------------------

    # ------------------------G's---------------------------------------------------------
    G1 = np.array([[*n1], [*t1]])
    G2 = np.array([[*n2], [*t2]])
    G3 = np.array([[*n3], [*t3]])
    B1 = G1_hat @ J_inv_T @ G1.T
    B2 = G2_hat @ J_inv_T @ G2.T
    B3 = G3_hat @ J_inv_T @ G3.T
    # ------------------------------------------------------------------------------------

    # ----------------------Theta---------------------------------------------------------
    THETA = np.array(
        [
            [P_1_x**2, 2 * P_1_x * P_2_x, P_2_x**2],
            [P_1_y * P_1_x, P_1_y * P_2_x + P_1_x * P_2_y, P_2_x * P_2_y],
            [P_1_y**2, 2 * P_1_y * P_2_y, P_2_y**2],
        ]
    )
    # ------------------------------------------------------------------------------------

    # -------------------------right-part-interp------------------------------------------
    right_part_interp = [
        *right_part_values[element[0]],
        *right_part_values[element[1]],
        *right_part_values[element[2]],
        n1 @ right_part_values[element[3]][1:3],
        n2 @ right_part_values[element[4]][1:3],
        n3 @ right_part_values[element[5]][1:3],
    ]
    # ------------------------------------------------------------------------------------

    # --------------------args------------------------------------------------------------
    args = [
        *n1,
        *n2,
        *n3,
        *t1,
        *t2,
        *t3,
        l1,
        l2,
        l3,
        *J.flatten(),
        *THETA.flatten(),
        *B1.flatten(),
        *B2.flatten(),
        *B3.flatten(),
        *right_part_interp,
    ]

    return args


def fill_stiffness_matrix(
    matrix, b, bilinear_form, right_part, element, vertex_marker_is_boundary, num_nodes
):
    for point_idx in range(3):
        if vertex_marker_is_boundary[element[point_idx]] == True:
            st_index = 3

            for i in range(0, st_index):
                matrix[6 * element[point_idx] + i, 6 * element[point_idx] + i] = 1

            for i in range(st_index, 6):
                for j in range(3):
                    for k in range(6):
                        matrix[6 * element[point_idx] + i, 6 * element[j] + k] += (
                            2 * bilinear_form[6 * point_idx + i, 6 * j + k]
                        )

                    for k in range(3, 6):
                        matrix[
                            6 * element[point_idx] + i,
                            6 * num_nodes + (element[k] - num_nodes),
                        ] += (
                            2 * bilinear_form[6 * point_idx + i, 18 + (k - 3)]
                        )

            for i in range(0, st_index):
                b[6 * element[point_idx] + i] = 0

            for i in range(st_index, 6):
                b[6 * element[point_idx] + i] += 2 * right_part[6 * point_idx + i]

        else:
            for i in range(6):
                for j in range(3):
                    for k in range(6):
                        matrix[6 * element[point_idx] + i, 6 * element[j] + k] += (
                            2 * bilinear_form[6 * point_idx + i, 6 * j + k]
                        )

                    for k in range(3, 6):
                        matrix[
                            6 * element[point_idx] + i,
                            6 * num_nodes + (element[k] - num_nodes),
                        ] += (
                            2 * bilinear_form[6 * point_idx + i, 18 + (k - 3)]
                        )

            for i in range(0, 6):
                b[6 * element[point_idx] + i] += 2 * right_part[6 * point_idx + i]

    for mid_idx in range(3, 6):
        if vertex_marker_is_boundary[element[mid_idx]] == True:
            matrix[
                6 * num_nodes + (element[mid_idx] - num_nodes),
                6 * num_nodes + (element[mid_idx] - num_nodes),
            ] = 1
            b[6 * num_nodes + (element[mid_idx] - num_nodes)] = 0
        else:
            for j in range(3):
                for k in range(6):
                    matrix[
                        6 * num_nodes + (element[mid_idx] - num_nodes),
                        6 * element[j] + k,
                    ] += (
                        2 * bilinear_form[18 + (mid_idx - 3), 6 * j + k]
                    )

                for k in range(3, 6):
                    matrix[
                        6 * num_nodes + (element[mid_idx] - num_nodes),
                        6 * num_nodes + (element[k] - num_nodes),
                    ] += (
                        2 * bilinear_form[18 + (mid_idx - 3), 18 + (k - 3)]
                    )
            b[6 * num_nodes + (element[mid_idx] - num_nodes)] += (
                2 * right_part[18 + (mid_idx - 3)]
            )


def get_delta_function(
    origin,
    points,
    triangles,
    num_nodes,
):
    integral_values = dill.load(
        open("../calculations/argyris_quintic_biharmonic_matrix_integral_values", "rb")
    )
    near_point = ((points[:num_nodes] - origin) ** 2).sum(axis=-1).argmin()
    base_triangles = np.where(near_point == triangles)
    area = 0

    for idx, element in enumerate(triangles[base_triangles[0]]):
        element = orient(element)
        trng = points[element]

        # ----------------P's----------------------------------------------------------------
        x1, x2, x3 = points[element[0], 0], points[element[1], 0], points[element[2], 0]
        y1, y2, y3 = points[element[0], 1], points[element[1], 1], points[element[2], 1]

        delta = x1 * y2 - x1 * y3 - x2 * y1 + x2 * y3 + x3 * y1 - x3 * y2

        P_1_x = (-y1 + y3) / delta
        P_1_y = (+x1 - x3) / delta

        P_2_x = (+y1 - y2) / delta
        P_2_y = (-x1 + x2) / delta
        # ------------------------------------------------------------------------------------

        # ----------------------Lengths-------------------------------------------------------
        l1 = ((x2 - x3) ** 2 + (y2 - y3) ** 2) ** 0.5
        l2 = ((x1 - x3) ** 2 + (y1 - y3) ** 2) ** 0.5
        l3 = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        # ------------------------------------------------------------------------------------

        # ----------------------Jacobian------------------------------------------------------
        J = np.array([[P_1_x, P_1_y], [P_2_x, P_2_y]])
        J_inv_T = np.linalg.inv(J).T
        # -----------------------------------------------------------------------------------

        # -----------------------nomal-and-tangential-vectors--------------------------------
        t1 = trng[2] - trng[1]
        t1 = t1 / np.linalg.norm(t1)
        n1 = rot @ t1

        t2 = trng[2] - trng[0]
        t2 = t2 / np.linalg.norm(t2)
        n2 = rot @ t2

        t3 = trng[1] - trng[0]
        t3 = t3 / np.linalg.norm(t3)
        n3 = rot @ t3
        # ------------------------------------------------------------------------------------

        # ------------------------G's---------------------------------------------------------
        G1 = np.array([[*n1], [*t1]])
        G2 = np.array([[*n2], [*t2]])
        G3 = np.array([[*n3], [*t3]])

        B1 = G1_hat @ J_inv_T @ G1.T
        B2 = G2_hat @ J_inv_T @ G2.T
        B3 = G3_hat @ J_inv_T @ G3.T
        # ------------------------------------------------------------------------------------

        # ----------------------Theta---------------------------------------------------------
        THETA = np.array(
            [
                [P_1_x**2, 2 * P_1_x * P_2_x, P_2_x**2],
                [P_1_y * P_1_x, P_1_y * P_2_x + P_1_x * P_2_y, P_2_x * P_2_y],
                [P_1_y**2, 2 * P_1_y * P_2_y, P_2_y**2],
            ]
        )
        # ------------------------------------------------------------------------------------

        # --------------------args------------------------------------------------------------
        args = [
            *n1,
            *n2,
            *n3,
            *t1,
            *t2,
            *t3,
            l1,
            l2,
            l3,
            *J.flatten(),
            *THETA.flatten(),
            *B1.flatten(),
            *B2.flatten(),
            *B3.flatten(),
        ]

        node_f_dof_idx = 6 * np.where(orient(element) == near_point)[0]

    area += integral_values(*args)[node_f_dof_idx]

    right_part_values = np.zeros((points.shape[0], 6))
    right_part_values[near_point][0] = 1 / area

    return right_part_values
