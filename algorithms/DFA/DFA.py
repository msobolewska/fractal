import numpy as np
from scipy.stats import linregress
from typing import List
import matplotlib.pyplot as plt


def divide_into_segments(data: List[float], segment_size: int) -> List[List[float]]:
    """
    :param data:
    :param segment_size:
    :return:
    """
    num_segments = len(data) // segment_size
    segments = np.array_split(data, num_segments)
    return segments


def calculate_local_trend(segment: List[int]) -> List[float]:
    """
    :param segment:
    :return:
    """
    x = np.arange(len(segment))
    slope, intercept, _, _, _ = linregress(x, segment)
    trend_line = slope * x + intercept
    print(x)
    print(segment)
    return trend_line


def dfa_process_single_segment(segment: List[float]) -> {float, float, float}:
    """
    :param segment:
    :return:
    """
    trend_line = calculate_local_trend(segment)
    detrended_segment = segment - trend_line
    segment_size = len(segment)
    variance = (1 / segment_size) * np.sum(detrended_segment ** 2)
    return variance, trend_line, detrended_segment


def dfa_process_single_segment_size(integrated_time_series: List[float],
                                    segment_size: int) -> float:
    """
    :param integrated_time_series:
    :param segment_size:
    :return:
    """
    segments = divide_into_segments(integrated_time_series, segment_size)
    variances = []

    for i, segment in enumerate(segments):
        variance, _, _ = dfa_process_single_segment(segment)
        variances.append(variance)

    average_var_fluctuation = np.sqrt(np.mean(variances))
    return average_var_fluctuation


def dfa_process_single_segment_size_first_x(integrated_time_series: List[int],
                                            segment_size: int,
                                            x: int) -> {float, List[float], List[float]}:
    """
    :param x:
    :param integrated_time_series:
    :param segment_size:
    :return:
    """
    segments = divide_into_segments(integrated_time_series, segment_size)
    variances = []
    trend_lines = []
    detrended_segments = []

    for i, segment in enumerate(segments):
        if i < x:
            variance, trend_line, detrended_segment = dfa_process_single_segment(segment)
            variances.append(variance)
            trend_lines.append(trend_line)
            detrended_segments.append(detrended_segment)
        else:
            break

    average_var_fluctuation = np.sqrt(np.mean(variances))
    return average_var_fluctuation, segments[0:x], trend_lines, detrended_segments


def dfa_segment_with_plot_x_parts(original_time_series: List[int],
                                  integrated_time_series: List[int],
                                  segment_size: int,
                                  x: int) -> None:
    """
    :param original_time_series:
    :param x:
    :param integrated_time_series:
    :param segment_size:
    :return:
    """
    _, segments, trend_lines, detrended_segmients = dfa_process_single_segment_size_first_x(
        integrated_time_series,
        segment_size, x)

    plot_segments(original_time_series, segments, trend_lines, detrended_segmients)


def get_h_from_fluctuations(segment_sizes: List[int], dfa_fluctuations: List[float]) -> float:
    """
    :param segment_sizes:
    :param dfa_fluctuations:
    :return:
    """
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

    dfa_fluctuations = []

    for ii, segment_size in enumerate(segment_sizes):
        average_var_fluctuation = dfa_process_single_segment_size(integrated_time_series,
                                                                  segment_size)
        dfa_fluctuations.append(average_var_fluctuation)

    estimated_a = get_h_from_fluctuations(segment_sizes, dfa_fluctuations)
    return estimated_a


def plot_segments(original_time_series: List[int], segments: List[List[float]], trend_lines: List[float],
                  detrended_segments: List[float]) -> None:
    """
    :param original_time_series:
    :param segments:
    :param trend_lines:
    :param detrended_segments:
    :return:
    """
    segments_flattened = [val for sublist in segments for val in sublist]
    len_segment_flattened = len(segments_flattened)
    x_axis = list(range(0, len_segment_flattened))

    fig, axs = plt.subplots(3)

    axs[0].scatter(x_axis, original_time_series[0:len_segment_flattened])
    axs[0].set_title('Time series')

    axs[1].scatter(x_axis, segments_flattened)

    segment_size = len(segments[0])
    no_segments = len(segments)

    for i in np.arange(no_segments):
        axs[1].plot(x_axis[i*segment_size: (i+1)*segment_size], trend_lines[i], color='red')

    axs[1].set_title('Integrated time series with trend')

    axs[2].scatter(x_axis, detrended_segments[0:len_segment_flattened])
    axs[2].set_title('Detrended time series')

    plt.tight_layout()
    plt.show()

