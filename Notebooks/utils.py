import numpy as np


def get_mid(a, b):
    if (a == 0) and (b == 1):
        return 5
    if (a == 1) and (b == 2):
        return 3
    if (a == 0) and (b == 2):
        return 4


def get_local_normals_and_tangentials(points, triangles):
    elem_normal = np.zeros((*triangles.shape, 2))
    elem_tangential = np.zeros((*triangles.shape, 2))

    for index, element in enumerate(triangles):
        elem_tangential[index, 0][0] = points[element[1]][0] - points[element[0]][0]
        elem_tangential[index, 0][1] = points[element[1]][1] - points[element[0]][1]

        elem_normal[index, 0][0] = points[element[1]][1] - points[element[0]][1]
        elem_normal[index, 0][1] = -(points[element[1]][0] - points[element[0]][0])

        elem_tangential[index, 1][0] = points[element[2]][0] - points[element[1]][0]
        elem_tangential[index, 1][1] = points[element[2]][1] - points[element[1]][1]

        elem_normal[index, 1][0] = points[element[2]][1] - points[element[1]][1]
        elem_normal[index, 1][1] = -(points[element[2]][0] - points[element[1]][0])

        elem_tangential[index, 2][0] = points[element[0]][0] - points[element[2]][0]
        elem_tangential[index, 2][1] = points[element[0]][1] - points[element[2]][1]

        elem_normal[index, 2][0] = points[element[0]][1] - points[element[2]][1]
        elem_normal[index, 2][1] = -(points[element[0]][0] - points[element[2]][0])

        mid1 = get_mid(0, 1)
        mid2 = get_mid(1, 2)
        mid3 = get_mid(0, 2)

        elem_normal[index, mid1][0] = -points[element[0]][1] + points[element[1]][1]
        elem_normal[index, mid1][1] = -(-points[element[0]][0] + points[element[1]][0])

        elem_tangential[index, mid1][0] = -points[element[0]][0] + points[element[1]][0]
        elem_tangential[index, mid1][1] = -points[element[0]][1] + points[element[1]][1]

        elem_normal[index, mid2][0] = -points[element[1]][1] + points[element[2]][1]
        elem_normal[index, mid2][1] = -(-points[element[1]][0] + points[element[2]][0])

        elem_tangential[index, mid2][0] = -points[element[1]][0] + points[element[2]][0]
        elem_tangential[index, mid2][1] = -points[element[1]][1] + points[element[2]][1]

        elem_normal[index, mid3][0] = -points[element[2]][1] + points[element[0]][1]
        elem_normal[index, mid3][1] = -(-points[element[2]][0] + points[element[0]][0])

        elem_tangential[index, mid3][0] = -points[element[2]][0] + points[element[0]][0]
        elem_tangential[index, mid3][1] = -points[element[2]][1] + points[element[0]][1]

    elem_normal = elem_normal / (elem_normal**2).sum(axis=-1, keepdims=True) ** 0.5
    elem_tangential = (
        elem_tangential / (elem_tangential**2).sum(axis=-1, keepdims=True) ** 0.5
    )

    return elem_normal, elem_tangential


def get_global_normals_and_tangentials(points, triangles):
    global_elem_normal = np.zeros((points.shape[0], 2))
    global_elem_tangential = np.zeros((points.shape[0], 2))
    is_set = np.zeros(points.shape[0], dtype=np.bool_)

    for index, element in enumerate(triangles):
        if is_set[element[0]] == False:
            global_elem_tangential[element[0]][0] = (
                points[element[1]][0] - points[element[0]][0]
            )
            global_elem_tangential[element[0]][1] = (
                points[element[1]][1] - points[element[0]][1]
            )

            global_elem_normal[element[0]][0] = (
                points[element[1]][1] - points[element[0]][1]
            )
            global_elem_normal[element[0]][1] = -(
                points[element[1]][0] - points[element[0]][0]
            )

            is_set[element[0]] = True

        if is_set[element[1]] == False:
            global_elem_tangential[element[1]][0] = (
                points[element[2]][0] - points[element[1]][0]
            )
            global_elem_tangential[element[1]][1] = (
                points[element[2]][1] - points[element[1]][1]
            )

            global_elem_normal[element[1]][0] = (
                points[element[2]][1] - points[element[1]][1]
            )
            global_elem_normal[element[1]][1] = -(
                points[element[2]][0] - points[element[1]][0]
            )

            is_set[element[1]] = True

        if is_set[element[2]] == False:
            global_elem_tangential[element[2]][0] = (
                points[element[0]][0] - points[element[2]][0]
            )
            global_elem_tangential[element[2]][1] = (
                points[element[0]][1] - points[element[2]][1]
            )

            global_elem_normal[element[2]][0] = (
                points[element[0]][1] - points[element[2]][1]
            )
            global_elem_normal[element[2]][1] = -(
                points[element[0]][0] - points[element[2]][0]
            )

            is_set[element[2]] = True

        if is_set[element[3]] == False:
            global_elem_normal[element[3]][0] = (
                -points[element[1]][1] + points[element[2]][1]
            )
            global_elem_normal[element[3]][1] = -(
                -points[element[1]][0] + points[element[2]][0]
            )

            global_elem_tangential[element[3]][0] = (
                -points[element[1]][0] + points[element[2]][0]
            )
            global_elem_tangential[element[3]][1] = (
                -points[element[1]][1] + points[element[2]][1]
            )

            is_set[element[3]] = True

        if is_set[element[4]] == False:
            global_elem_normal[element[4]][0] = (
                -points[element[2]][1] + points[element[0]][1]
            )
            global_elem_normal[element[4]][1] = -(
                -points[element[2]][0] + points[element[0]][0]
            )

            global_elem_tangential[element[4]][0] = (
                -points[element[2]][0] + points[element[0]][0]
            )
            global_elem_tangential[element[4]][1] = (
                -points[element[2]][1] + points[element[0]][1]
            )

            is_set[element[4]] = True

        if is_set[element[5]] == False:
            global_elem_normal[element[5]][0] = (
                -points[element[0]][1] + points[element[1]][1]
            )
            global_elem_normal[element[5]][1] = -(
                -points[element[0]][0] + points[element[1]][0]
            )

            global_elem_tangential[element[5]][0] = (
                -points[element[0]][0] + points[element[1]][0]
            )
            global_elem_tangential[element[5]][1] = (
                -points[element[0]][1] + points[element[1]][1]
            )

            is_set[element[5]] = True

    global_elem_normal = (
        global_elem_normal
        / (global_elem_normal**2).sum(axis=-1, keepdims=True) ** 0.5
    )
    global_elem_tangential = (
        global_elem_tangential
        / (global_elem_tangential**2).sum(axis=-1, keepdims=True) ** 0.5
    )

    return global_elem_normal, global_elem_tangential


def get_middle_indices(num_points, triangles):
    is_middle = np.zeros(num_points, dtype=np.bool_)
    for idx, elem in enumerate(triangles):
        is_middle[elem[3:]] = True

    return is_middle


def get_delta_function(
    origin,
    points,
    triangles,
    elem_normal,
    elem_tangential,
    global_elem_normal,
    global_elem_tangential,
    num_nodes,
    basis_integral_values,
):
    near_point = ((points[:num_nodes] - origin) ** 2).sum(axis=-1).argmin()
    base_triangles = np.where(near_point == triangles)

    area = 0

    for idx, element in enumerate(triangles[base_triangles[0]]):
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

        J_inv_T = np.linalg.inv(J.T)
        # -----------------------------------------------------------------------------------

        # -----------------------nomal-and-tangential-vectors--------------------------------
        n1 = elem_normal[idx, 3]
        t1 = elem_tangential[idx, 3]

        n2 = elem_normal[idx, 4]
        t2 = elem_tangential[idx, 4]

        n3 = elem_normal[idx, 5]
        t3 = elem_tangential[idx, 5]

        DN_n1 = global_elem_normal[triangles[idx]][3]
        DN_t1 = global_elem_tangential[triangles[idx]][3]

        DN_n2 = global_elem_normal[triangles[idx]][4]
        DN_t2 = global_elem_tangential[triangles[idx]][4]

        DN_n3 = global_elem_normal[triangles[idx]][5]
        DN_t3 = global_elem_tangential[triangles[idx]][5]
        # ------------------------------------------------------------------------------------

        # ------------------------G's---------------------------------------------------------
        G1 = np.array([[*n1], [*t1]])

        G2 = np.array([[*n2], [*t2]])

        G3 = np.array([[*n3], [*t3]])

        G1_hat = np.array(
            [
                [+np.sqrt(1 / 2), +np.sqrt(1 / 2)],
                [-np.sqrt(1 / 2), +np.sqrt(1 / 2)],
            ]
        )

        G2_hat = np.array(
            [
                [-1, +0],
                [+0, -1],
            ]
        )

        G3_hat = np.array([[+0, -1], [+1, +0]])

        B1 = G1_hat @ J_inv_T @ G1.T
        B2 = G2_hat @ J_inv_T @ G2.T
        B3 = G3_hat @ J_inv_T @ G3.T

        DN_G1 = np.array([[*DN_n1], [*DN_t1]])

        DN_G2 = np.array([[*DN_n2], [*DN_t2]])

        DN_G3 = np.array([[*DN_n3], [*DN_t3]])

        DN_B1 = G1_hat @ J_inv_T @ DN_G1.T
        DN_B2 = G2_hat @ J_inv_T @ DN_G2.T
        DN_B3 = G3_hat @ J_inv_T @ DN_G3.T
        # ------------------------------------------------------------------------------------

        # ----------------------Theta---------------------------------------------------------
        a11 = -x1 + x2
        a12 = -x1 + x3

        a21 = -y1 + y2
        a22 = -y1 + y3

        THETA = np.array(
            [
                [a11**2, 2 * a11 * a21, a21**2],
                [a12 * a11, a12 * a21 + a11 * a22, a21 * a22],
                [a12**2, 2 * a12 * a22, a22**2],
            ]
        )
        # ------------------------------------------------------------------------------------

        # -------------------------right-part-interp------------------------------------------

        right_part_interp = [
            1,
        ] * 21
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
            *DN_B1.flatten(),
            *DN_B2.flatten(),
            *DN_B3.flatten(),
            *right_part_interp,
        ]

        area += basis_integral_values(*args)[base_triangles[1][idx]]

    right_part_values = np.zeros((points.shape[0], 6))
    right_part_values[near_point][0] = 1 / area

    return right_part_values
