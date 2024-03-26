import logging
import numpy as np
from typing import List
from scipy.stats import linregress

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
    d_u_t_s = X[t + u - 1] - X[t - 1] - (u / s) * (X[t + s - 1] - X[t - 1])
    return d_u_t_s


def calc_cumulated_range(X: List[float], t: int, s: int) -> float:
    """
    :param X:
    :param t:
    :param s:
    :return:
    """
    u = range(0, s + 1)
    d_u_t_s = [define_detrended_subrecord(uu, t, s, X) for uu in u]
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
    var_p1 = (1 / s) * np.sum([ksi[t + w - 1] * ksi[t + w - 1]] for w in range(1, s + 1))
    var_p2 = (1 / s) * np.sum([ksi[t + w - 1] for w in range(1, s + 1)])
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

    slope, intercept, _, _, _ = linregress(log_segment_values, log_F_values)

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

    for ss, pp in s, p:
        t = define_starting_points(ss, pp)
        ratios_per_s = []
        for tt in t:
            r_t_s = calc_cumulated_range(X, tt, ss)
            variance_squared = calc_variance_squared(tt, ss, ksi)
            ratios_per_s.append(r_t_s / (np.sqrt(variance_squared)))
        ratios.append(np.mean(ratios_per_s))

    estimated_a = get_h_from_fluctuations(s, ratios)
    return estimated_a
