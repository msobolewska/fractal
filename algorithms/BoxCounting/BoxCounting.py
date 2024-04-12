from typing import List

import numpy as np
from scipy.stats import linregress


def divide_into_segments(data: List[float], segment_size: int) -> List[List[float]]:
    """
    :param data:
    :param segment_size:
    :return:
    """
    num_segments = len(data) // segment_size
    segments = np.array_split(data, num_segments)
    return segments


def boxcounting_process_single_segment_size(time_series: List[float],
                                            segment_size: int) -> List[float]:
    """
    :param time_series:
    :param segment_size:
    :return:
    """
    segments = divide_into_segments(time_series, segment_size)
    sum_total = sum(time_series)
    p_s_v_segments = []

    for i, segment in enumerate(segments):
        p_s_v = sum(segment) / sum_total
        p_s_v_segments.append(p_s_v)

    return p_s_v_segments


def boxcounting_process_single_q_value(p_s_v_segments: List[float],
                                       q: int) -> float:
    """
    :param p_s_v_segments:
    :param q:
    :return:
    """
    chi = sum(x ** q for x in p_s_v_segments)
    return chi


def boxcounting_process_q_values(p_s_v_segments: List[float],
                                 qs: List[int]) -> {List[int], List[float]}:
    """
    :param p_s_v_segments:
    :param qs:
    :return:
    """
    chis = [boxcounting_process_single_q_value(p_s_v_segments, q) for q in qs]
    return qs, chis


def get_tau_from_ksis(segment_sizes: List[int], chis: List[float]) -> float:
    """
    :param segment_sizes:
    :param chis:
    :return:
    """
    log_segment_sizes = np.log(segment_sizes)
    log_chis = np.log(chis)

    slope, intercept, _, _, _ = linregress(log_segment_sizes, log_chis)

    estimated_tau = slope
    return estimated_tau


def update_qs_ksis_over_segments(qs_ksis_over_segments: {}, x_value: int, new_y_values: List[float]):
    if x_value in qs_ksis_over_segments:
        qs_ksis_over_segments[x_value][0].append(new_y_values[0])
        qs_ksis_over_segments[x_value][1].append(new_y_values[1])
    else:
        qs_ksis_over_segments[x_value] = [new_y_values[:1], new_y_values[1:]]


def calculate_fractal_dimension(qs: List[int], estimated_taus: List[float]) -> List[float]:
    """
    :param qs:
    :param estimated_taus:
    :return:
    """
    ds = [estimated_tau / (q - 1) for estimated_tau, q in zip(estimated_taus, qs)]
    return ds


def perform_boxcounting(length_time_series: List[int],
                        segment_sizes: List[int], qs: List[int]) -> {List[int], List[float]}:
    """
    :param length_time_series:
    :param segment_sizes:
    :param qs:
    :return:
    """
    qs_ksis_over_segments = {}
    estimated_taus = []

    for ii, segment_size in enumerate(segment_sizes):
        p_c_v_segments = boxcounting_process_single_segment_size(length_time_series,
                                                                 segment_size)
        qs, ksis = boxcounting_process_q_values(p_c_v_segments, qs)

        for qss, ksiss in zip(qs, ksis):
            update_qs_ksis_over_segments(qs_ksis_over_segments, qss, [segment_size, ksiss])

    for qss in qs:
        values = qs_ksis_over_segments[qss]
        segment_sizes = values[0]
        ksis = values[1]
        estimated_tau = get_tau_from_ksis(segment_sizes, ksis)
        estimated_taus.append(estimated_tau)

    ds = calculate_fractal_dimension(qs, estimated_taus)
    return qs, ds
