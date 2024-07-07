from typing import List, Tuple, Any

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray

from algorithms.DFA.DFA import get_h_from_fluctuations


def divide_into_overlapping_segments(data: List[float], w: int) -> List[List[float]]:
    """
    :param data:
    :param w:
    :return:
    """
    overlap = (w - 1) // 2 + 1

    segments = []

    for i in range(0, len(data), w - overlap):
        segments.append(data[i:min(i + w, len(data))])
        # last segment requires special treatment, this probably can be improved by cleaner code
        if i + w > len(data):
            break

    # print(segments[:10])
    return segments


def calculate_coefficients_of_local_trend_order_n(segment: List[float], n: int = 1) -> {ndarray, Any, ndarray}:
    """
    :param n:
    :param segment:
    :return:
    """
    x = np.arange(len(segment))
    coefficients = np.polyfit(x, segment, n)
    return coefficients


def afa_process_segments(segments: List[List[float]]) -> {List[float], Any}:
    """
    :param segments:
    :return:
    """
    coefficients = [calculate_coefficients_of_local_trend_order_n(segment, 2) for segment in segments]
    w = len(segments[0])
    overlap = (w - 1) // 2 + 1
    n = overlap - 1
    x = np.arange(w)

    n_segments = len(segments)

    global_trend = []

    current_index = 0

    for i in range(n_segments - 1):
        # print('now is segment number ', i, ' of total ', n_segments)
        coeff_i = coefficients[i]
        y_i = np.poly1d(coeff_i)
        fit_values_i = y_i(x)

        coeff_i_plus_1 = coefficients[i + 1]
        y_i_plus_1 = np.poly1d(coeff_i_plus_1)
        fit_values_i_plus_1 = y_i_plus_1(x)

        fit_of_overlapped = [(1 - l / n) * fit_values_i[l + n] + (l / n) * fit_values_i_plus_1[l] for l in
                             range(0, n + 1)]

        global_trend.extend(fit_values_i[current_index: w - overlap])
        global_trend.extend(fit_of_overlapped)
        current_index += w

        # for last segment, append fitted values from corresponding segment trend line up until end of segment
        if i == n_segments - 2:
            global_trend.extend(fit_values_i_plus_1[n + 1:len(segments[-1])])

    return global_trend, coefficients


def afa_process_single_segment_size(integrated_time_series: List[float],
                                    w: int) -> float:
    """
    :param integrated_time_series:
    :param w:
    :return:
    """

    segments = divide_into_overlapping_segments(integrated_time_series, w)

    global_trend, _ = afa_process_segments(segments)

    variances = [(a1 - a2) ** 2 for a1, a2 in zip(integrated_time_series, global_trend)]

    average_var_fluctuation = np.sqrt(np.mean(variances))
    return average_var_fluctuation


def perform_afa(integrated_time_series: List[int],
                ws: List[int]) -> float:
    """
    :param ws:
    :param integrated_time_series:
    :return:
    """

    afa_fluctuations = []
    ws_nona = []  # store lengths that return non-nan; nan occurs for example when length of segments is greater that
    # actual list length

    for ii, w in enumerate(ws):
        average_var_fluctuation = afa_process_single_segment_size(integrated_time_series,
                                                                  w)
        if not np.isnan(average_var_fluctuation):
            afa_fluctuations.append(average_var_fluctuation)
            ws_nona.append(w)
        else:
            print(f' NaN encountered for segment length: {w}')

    if len(afa_fluctuations) == 0:
        print(f'No non-nans encountered for segment')
        return np.nan
    #print(f' afa_fluctuations: {afa_fluctuations}')
    estimated_a = get_h_from_fluctuations(ws_nona, afa_fluctuations)
    return estimated_a


# for plotting purposes below method was created

def plot_segments_x(original_time_series: List[int], integrated_time_series: List[float], w: int = 20,
                    x: int = 3) -> None:
    """
    :param w:
    :param integrated_time_series:
    :param x:
    :param original_time_series:
    :return:
    """
    segments_overlapped = divide_into_overlapping_segments(integrated_time_series, w)
    segments_overlapped = segments_overlapped[:x]

    overlap = (w - 1) // 2 + 1

    global_trend, coefficients = afa_process_segments(segments_overlapped)

    len_segments_nonoverlapped = x * w - (x - 1) * overlap

    x_axis = list(range(0, len_segments_nonoverlapped))

    fig, axs = plt.subplots(3)  # TODO podpisac osie

    axs[0].scatter(x_axis, original_time_series[0:len_segments_nonoverlapped])
    axs[0].set_title('Time series')
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('u_t')

    axs[1].scatter(x_axis, integrated_time_series[0:len_segments_nonoverlapped])

    xx = np.arange(w)

    # TODO make it more general, so x is used
    axs[1].plot(x_axis[0: w], np.poly1d(coefficients[0])(xx), color='red')
    axs[1].plot(x_axis[w - overlap: 2 * w - overlap], np.poly1d(coefficients[1])(xx), color='green')
    axs[1].plot(x_axis[2 * w - 2 * overlap: 3 * w - 2 * overlap], np.poly1d(coefficients[2])(xx), color='red')
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('Y(t)')

    axs[1].set_title('Integrated time series with trend')

    print('length of x_axis ', len(x_axis))
    print('length of global_trend[0:len_segments_nonoverlapped] ', len(global_trend[0:len_segments_nonoverlapped]))
    axs[2].scatter(x_axis, integrated_time_series[0:len_segments_nonoverlapped])
    axs[2].plot(x_axis, global_trend[0:len_segments_nonoverlapped], color='red')
    axs[2].set_title('Global trend')
    axs[2].set_xlabel('t')
    axs[2].set_ylabel('Y(t)')

    plt.tight_layout()
    plt.savefig("afa.png")
    plt.show()
