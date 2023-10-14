import dill
import numpy as np


def get_p3_from_p1(
    triangles, edges, points, vertex_marker_is_boundary, edge_marker_is_boundary
):
    def get_local_edges(points, triangle, sorted_edges):
        result = []

        for pair in triangle[np.array([[1, 2], [0, 2], [0, 1]])]:
            pairs_sorted_idx = np.argsort(pair)
            idx = np.where(np.all(pair[pairs_sorted_idx] == sorted_edges, axis=-1))[0][
                0
            ]
            result.append(idx)

        return result

    extended_triangles = np.zeros((triangles.shape[0], 10), dtype=int)
    triangle_to_edge = np.zeros_like(triangles)

    edges_sorted_idx = np.argsort(edges, axis=-1)
    sorted_edges = np.take_along_axis(edges, edges_sorted_idx, axis=-1)

    # associate edges with triangles
    for tidx, triangle in enumerate(triangles):
        result = get_local_edges(points, triangle, sorted_edges)
        triangle_to_edge[tidx] = result

    num_nodes = points.shape[0]

    trisection_points = []

    extended_points = points.copy()
    extended_vmisb = vertex_marker_is_boundary.copy()

    for eidx, edge in enumerate(edges):
        m1 = (
            extended_points[edge[0]]
            + (-extended_points[edge[0]] + extended_points[edge[1]]) * 1 / 3
        )
        m2 = (
            extended_points[edge[0]]
            + (-extended_points[edge[0]] + extended_points[edge[1]]) * 2 / 3
        )

        extended_points = np.concatenate([extended_points, [m1, m2]])
        trisection_points.append(
            np.array([num_nodes + 2 * eidx, num_nodes + 2 * eidx + 1])
        )

        if edge_marker_is_boundary[eidx] == 1:
            extended_vmisb = np.concatenate([extended_vmisb, [[1], [1]]])
        else:
            extended_vmisb = np.concatenate([extended_vmisb, [[0], [0]]])

    trisection_points = np.stack(trisection_points)

    for tidx, triangle in enumerate(triangles):
        extended_triangles[tidx][:3] = triangle

        res = []

        # add add trisection points to the new triangles
        for global_edge, local_edge in zip(
            triangle_to_edge[tidx], triangle[np.array([[1, 2], [0, 2], [0, 1]])]
        ):
            what = trisection_points[global_edge]
            # check the case when the directions do not match
            if not np.all(edges[global_edge] == local_edge):
                what = what[::-1]

            res.extend(what.tolist())

        extended_triangles[tidx][3:-1] = res

    num_nodes = extended_points.shape[0]

    # add middle point
    for tidx, triangle in enumerate(extended_triangles):
        mid_point = extended_points[triangle[:3]].mean(axis=0)
        extended_points = np.concatenate([extended_points, [mid_point]])
        triangle[-1] = num_nodes + tidx
        extended_vmisb = np.concatenate([extended_vmisb, [[0]]])

    return extended_points, extended_triangles, extended_vmisb


def orient_batch(arg):
    indices = np.argsort(arg, axis=-1)
    oriented = np.take_along_axis(arg, indices, axis=-1)
    return oriented


def get_middle_indices(num_points, triangles):
    is_middle = np.zeros(num_points, dtype=np.bool_)
    for idx, elem in enumerate(triangles):
        is_middle[elem[3:]] = True

    return is_middle


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


def get_delta_approximation_argyris_biharmonic(
    origin, points, triangles, num_nodes, total_points
):
    integral_values = dill.load(
        open(
            "../calculations/argyris_quintic_biharmonic_matrix_integral_values_simplified",
            "rb",
        )
    )
    near_point = ((points[:num_nodes] - origin) ** 2).sum(axis=-1).argmin()
    base_triangles = np.where(near_point == triangles)

    pts = points[triangles[base_triangles[0], :3]]

    all_volumes = integral_values(
        pts[:, 0, 0],
        pts[:, 0, 1],
        pts[:, 1, 0],
        pts[:, 1, 1],
        pts[:, 2, 0],
        pts[:, 2, 1],
    )[:, 0].T

    total_volume = all_volumes[np.arange(pts.shape[0]), 6 * base_triangles[1]].sum()

    right_part_values = np.zeros((total_points, 6))
    right_part_values[near_point][0] = 1 / total_volume

    return right_part_values


def get_delta_approximation_bell_biharmonic(origin, points, triangles, total_points):
    integral_values = dill.load(
        open(
            "../calculations/bell_quintic_biharmonic_matrix_integral_values_simplified",
            "rb",
        )
    )
    near_point = ((points - origin) ** 2).sum(axis=-1).argmin()
    base_triangles = np.where(near_point == triangles)

    pts = points[triangles[base_triangles[0]]]

    all_volumes = integral_values(
        pts[:, 0, 0],
        pts[:, 0, 1],
        pts[:, 1, 0],
        pts[:, 1, 1],
        pts[:, 2, 0],
        pts[:, 2, 1],
    )[:, 0].T

    total_volume = all_volumes[np.arange(pts.shape[0]), 6 * base_triangles[1]].sum()

    right_part_values = np.zeros((total_points, 6))
    right_part_values[near_point][0] = 1 / total_volume

    return right_part_values


def fill_stiffness_matrix_bell_preconditioned(
    matrix, b, bilinear_form, right_part, element, vertex_marker_is_boundary, cond
):
    for point_idx in range(3):
        if vertex_marker_is_boundary[element[point_idx]] == True:
            st_index = 3

            for i in range(0, st_index):
                I = 6 * element[point_idx] + i
                J = 6 * element[point_idx] + i

                matrix[I, J] = 1
                b[I] = 0

            for i in range(st_index, 6):
                for j in range(3):
                    for k in range(6):
                        I = 6 * element[point_idx] + i
                        J = 6 * element[j] + k

                        value = 2 * bilinear_form[6 * point_idx + i, 6 * j + k]

                        p1 = 0 if i == 0 else 1 if (1 <= i <= 2) else 2
                        p2 = 0 if k == 0 else 1 if (1 <= k <= 2) else 2

                        value /= cond[element[point_idx]] ** p1
                        value /= cond[element[j]] ** p2

                        matrix[I, J] += value

            for i in range(st_index, 6):
                value = 2 * right_part[6 * point_idx + i]

                p1 = 0 if i == 0 else 1 if (1 <= i <= 2) else 2
                value /= cond[element[point_idx]] ** p1

                b[6 * element[point_idx] + i] += value

        else:
            for i in range(6):
                for j in range(3):
                    for k in range(6):
                        I = 6 * element[point_idx] + i
                        J = 6 * element[j] + k

                        value = 2 * bilinear_form[6 * point_idx + i, 6 * j + k]

                        p1 = 0 if i == 0 else 1 if (1 <= i <= 2) else 2
                        p2 = 0 if k == 0 else 1 if (1 <= k <= 2) else 2

                        value /= cond[element[point_idx]] ** p1
                        value /= cond[element[j]] ** p2

                        matrix[I, J] += value

            for i in range(6):
                value = 2 * right_part[6 * point_idx + i]

                p1 = 0 if i == 0 else 1 if (1 <= i <= 2) else 2
                value /= cond[element[point_idx]] ** p1

                b[6 * element[point_idx] + i] += value


def get_precondition_terms(points, triangles):
    tmp = points[triangles]
    v1 = tmp[:, 1] - tmp[:, 0]
    v2 = tmp[:, 2] - tmp[:, 0]
    v3 = tmp[:, 2] - tmp[:, 1]

    # get areas
    areas = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]) / 2
    P = (
        (v1**2).sum(axis=-1) ** 0.5
        + (v2**2).sum(axis=-1) ** 0.5
        + (v3**2).sum(axis=-1) ** 0.5
    )

    points_indices = np.arange(points.shape[0])

    vertex_1 = np.where(points_indices[:, None] == triangles[:, 0])
    vertex_2 = np.where(points_indices[:, None] == triangles[:, 1])
    vertex_3 = np.where(points_indices[:, None] == triangles[:, 2])

    map_vertex = np.concatenate([vertex_1[0], vertex_2[0], vertex_3[0]])
    map_triangle = np.concatenate([vertex_1[1], vertex_2[1], vertex_3[1]])

    cond = np.zeros(points.shape[0])

    for p_index in points_indices:
        w = 2 * areas[map_triangle[map_vertex == p_index]]
        w = w / P[map_triangle[map_vertex == p_index]]
        w = w.mean()
        cond[p_index] = w

    return cond


def is_extreme_boundary(edges, points, edge_marker_is_boundary, point_index):
    containing_edges, positions = np.where(edges == point_index)
    mask = edge_marker_is_boundary[containing_edges, 0].astype(bool)
    containing_edges = containing_edges[mask]

    point_coordinates = points[edges[containing_edges]]
    V = point_coordinates[..., 1, :] - point_coordinates[..., 0, :]

    v = V[0]
    v = v / np.linalg.norm(v)
    is_exptreme = ~np.isclose(np.linalg.det(V), 0)
    return is_exptreme, v
