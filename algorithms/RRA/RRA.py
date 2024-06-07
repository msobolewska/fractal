import logging
import numpy as np
from typing import List
from scipy.stats import linregress
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def find_m_between_T(T: int) -> int:
    """
    :param T:
    :return:
    """
    m = 0
    while True:
        if 2 ** m <= T <= 2 ** (m + 1):
            logger.info("For T equal to {}, m is {}", T, m)
            return m
        m += 1


def define_collection_s(T: int, m: int) -> {List[int], List[int]}:
    """
    :param T:
    :param m:
    :return:
    """
    p = range(0, m - 1)
    s = [T // (2 ** pp) for pp in p]
    return s, p


def define_starting_points(s: int, p: int) -> List[int]:
    """
    :param p:
    :param s:
    :return:
    """
    t = [q * s + 1 for q in range(0, 2 ** p)]
    return t


def define_detrended_subrecord(u: int, t: int, s: int, X: List[float]) -> float:
    """
    :param X:
    :param u:
    :param t:
    :param s:
    :return:
    """
    T = len(X)
    end = min(t + s, T)
    length = end - t
    d_u_t_s = X[t + u - 1] - X[t - 1] - (u / length) * (X[end - 1] - X[t - 1])
    return d_u_t_s


def calc_cumulated_range(X: List[float], t: int, s: int) -> float:
    """
    :param X:
    :param t:
    :param s:
    :return:
    """
    u = range(0, s + 1)
    T = len(X)
    d_u_t_s = [define_detrended_subrecord(uu, t, s, X) for uu in u if t + uu - 1 < T]
    r_t_s = max(d_u_t_s) - min(d_u_t_s)
    logger.info("r_t_s is {}", r_t_s)
    return r_t_s


def calc_variance_squared(t: int, s: int, ksi: List[float]) -> float:
    """
    :param t:
    :param s:
    :param ksi:
    :return:
    """
    T = len(ksi)
    var_p1 = (1 / s) * sum([ksi[t + w - 1] * ksi[t + w - 1] for w in range(1, s + 1) if t + w - 1 < T])
    var_p2 = (1 / s) * sum([ksi[t + w - 1] for w in range(1, s + 1) if t + w - 1 < T])
    var = var_p1 - var_p2 * var_p2
    return var


def get_h_from_fluctuations(segment_sizes: List[int], dfa_fluctuations: List[float]) -> float:
    """
    :param segment_sizes:
    :param dfa_fluctuations:
    :return:
    """
    log_segment_values = np.log(segment_sizes)
    log_F_values = np.log(dfa_fluctuations)

    if len(log_segment_values) == 0:
        print(f'returning None from get_h_from_fluctuations, segment_sizes: {segment_sizes}, log_segment_values: {log_segment_values}')
        return None

    slope, intercept, _, _, _ = linregress(log_segment_values, log_F_values)

    #TODO move it to the print method
    regression_line = slope * log_segment_values + intercept

    # Plot data and regression line
    #plt.scatter(log_segment_values, log_F_values, label='Data')
    #plt.plot(log_segment_values, regression_line, color='red', label='Regression Line')
    #plt.xlabel('Log(s)')
    #plt.ylabel('Log(R/S)')
    #plt.title('Linear Regression')
    #plt.legend()
    #plt.grid(True)
    #plt.savefig('rra.png')
    #plt.show()

    # Display slope and intercept
    #print("Slope:", slope)
    #print("Intercept:", intercept)

    estimated_h = slope
    return estimated_h


def perform_rra(X: List[int], ksi: List[float]) -> float:
    """
    :param X:
    :param ksi:
    :return:
    """
    T = len(X)
    m = find_m_between_T(T)
    s, p = define_collection_s(T, m)

    ratios = []

    for ss, pp in zip(s, p):
        t = define_starting_points(ss, pp)
        ratios_per_s = []
        for tt in t:
            r_t_s = calc_cumulated_range(X, tt, ss)
            variance_squared = calc_variance_squared(tt, ss, ksi)
            if variance_squared > 0:
                ratios_per_s.append(r_t_s / (np.sqrt(variance_squared)))
        ratios.append(np.mean(ratios_per_s))

    estimated_a = get_h_from_fluctuations(s, ratios)
    return estimated_a
