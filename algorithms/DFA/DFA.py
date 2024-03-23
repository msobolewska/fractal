import numpy as np
from scipy.stats import linregress
from typing import List


def divide_into_segments(data: List[int], segment_size: int):
    """
    :param data:
    :param segment_size:
    :return:
    """
    num_segments = len(data) // segment_size
    segments = np.array_split(data, num_segments)
    return segments


def calculate_local_trend(segment: List[int]):
    """
    :param segment:
    :return:
    """
    x = np.arange(len(segment))
    slope, intercept, _, _, _ = linregress(x, segment)
    trend_line = slope * x + intercept
    return trend_line


def dfa_process_single_segment(segment: List[int]) -> float:
    """
    :param segment:
    :return:
    """
    trend_line = calculate_local_trend(segment)
    detrended_segment = segment - trend_line
    segment_size = len(segment)
    variance = (1 / segment_size) * np.sum(detrended_segment ** 2)
    return variance


def dfa_process_single_segment_size(integrated_time_series: List[int],
                                    segment_size: int) -> float:
    """
    :param integrated_time_series:
    :param segment_size:
    :return:
    """
    segments = divide_into_segments(integrated_time_series, segment_size)
    variances = []

    for i, segment in enumerate(segments):
        variance = dfa_process_single_segment(segment)
        variances.append(variance)

    average_var_fluctuation = np.sqrt(np.mean(variances))
    # segment_sizes.append(segment_size)
    return average_var_fluctuation


def get_h_from_fluctuations(segment_sizes: List[int], dfa_fluctuations: List[float]) -> float:
    log_segment_values = np.log(segment_sizes)
    log_F_values = np.log(dfa_fluctuations)

    slope, intercept, _, _, _ = linregress(log_segment_values, log_F_values)

    estimated_h = slope
    return estimated_h


def perform_dfa(integrated_time_series: List[int],
                segment_sizes: List[int]) -> float:
    """
    :param integrated_time_series:
    :param segment_sizes: List of sizes into which series is split
    :return:
    """

    #segment_sizes = []
    dfa_fluctuations = []

    for ii, segment_size in enumerate(segment_sizes):
        average_var_fluctuation = dfa_process_single_segment_size(integrated_time_series,
                                                                  segment_size)
        dfa_fluctuations.append(average_var_fluctuation)

    estimated_a = get_h_from_fluctuations(segment_sizes, dfa_fluctuations)
    #print(estimated_a)
    return estimated_a
