from collections.abc import Sequence
import numpy as np
import math


def next_power_of_two(n_points):
    """ returns the next power of 2 for FFT, n_points is n the number of points. If n_points is 10, the function
    returns 16.
    """
    next_power = 1
    while next_power < n_points:
        next_power *=2
    return next_power


def max_min(curve, first_pt, last_pt):
    max_location = min_location = 0
    max = min = curve[first_pt]
    for i in range(first_pt, last_pt):
        if curve[i] < min:
            min = curve[i]
            min_location = i
        if curve[i] > max:
            max = curve[i]
            max_location = i

    return max, min, max_location, min_location


def simpson_nonuniform(x: Sequence[float], f: Sequence[float]) -> float:
    """
    Simpson rule for irregularly spaced data.

    :param x: Sampling points for the function values
    :param f: Function values at the sampling points

    :return: approximation for the integral

    See ``scipy.integrate.simpson`` and the underlying ``_basic_simpson``
    for a more performant implementation utilizing numpy's broadcast.
    """
    N = len(x) - 1
    h = [x[i + 1] - x[i] for i in range(0, N)]
    assert N > 0

    result = 0.0
    for i in range(1, N, 2):
        h0, h1 = h[i - 1], h[i]
        hph, hdh, hmh = h1 + h0, h1 / h0, h1 * h0
        result += (hph / 6) * (
            (2 - hdh) * f[i - 1] + (hph**2 / hmh) *
            f[i] + (2 - 1 / hdh) * f[i + 1]
        )

    if N % 2 == 1:
        h0, h1 = h[N - 2], h[N - 1]
        result += f[N] * (2 * h1 ** 2 + 3 * h0 * h1) / (6 * (h0 + h1))
        result += f[N - 1] * (h1 ** 2 + 3 * h1 * h0) / (6 * h0)
        result -= f[N - 2] * h1 ** 3 / (6 * h0 * (h0 + h1))
    return result


def simpsons_rule(curve, first_pt, last_pt, dt):
    """
    Simpson's rule numerical integration for regularly spaced data using Simpson's 1/3 rule.
    Automatically uses Simpson's 3/8 rule for even number of points
    :param curve: array to be integrated, time values must be equally spaced.
    :param first_pt: integer of first point in array curve to begin integration.
    :param last_pt: integer of last point in array curve to end integration.
    :param dt: time between points, typically 1.0 / sampling rate.
    :return: area as a float point number.
    see bottom of this wiki link for Python irregularly spaced code
    https://en.wikipedia.org/wiki/Simpson%27s_rule
    """
    area = 0.0
    n_pts = last_pt - first_pt
    if n_pts > 2:
        if (n_pts % 2) != 0:                                        # section for odd number of points 1/3 rule
            for i in range(first_pt, last_pt - 1, 2):
                area += dt * (curve[i] + 4 * curve[i + 1] + curve[i + 2]) / 3
        else:                                                       # section for even number of points 3/8's rule
            area += 3 * dt * (
                        curve[first_pt] + 3 * curve[first_pt + 1] + 3 * curve[first_pt + 2] + curve[first_pt + 3]) / 8.0
            for i in range(first_pt + 3, last_pt - 1, 2):
                area += dt * (curve[i] + 4 * curve[i + 1] + curve[i + 2]) / 3.0
    elif n_pts == 2:                                        # only two points use Midpoint rule
        area = dt * (curve[last_pt] + curve[first_pt]) / 2.0
    else:                                                   # only one point use Rectangle rule
        area = dt * curve[first_pt]
    return area


def zero_crossing(curve, reference_value, start, stop):
    """Finds all locations where the values in curve[] cross or are equal to the reference value.
    Args:
        curve (ndarray): numpy array
        reference_value (float): value to search for in array curve[]
        start (int): first point to begin searching for reference value in curve[]
        stop (int): last point to end the search for reference value in curve[]. Usually n - 1
    Returns:
        zlist: a list containing the indexes to the locations in curve[]
        where the values cross or are equal to the reference_value.
        rise_or_fall: a list containing the direction 'rising' or 'falling'
        for each index point found in the input array curve[].
    example 1:
    x = np.array([0, -1.1, .2, 3.2, 2.9, .8, 0.0, -.7, -.2, 0])
    crosspoints = zero_crossing(x, 0, 0, 9)  # find all 0.0's
    returns:
    zlist = [0, 1, 6, 9]
    example 2:
    x = np.array([0, -1.1, .2, 3.2, 2.9, .8, 0.0, -.7, -.2, 0])
    crosspoints = zero_crossing(x, 0.2, 0, 9)  # find all 0.2's
    returns:
    zlist = [2, 5]
    """
    zlist = []              # list to hold the indexes of curve[i] = reference_value
    if stop > len(curve) - 1:
        stop = len(curve) - 1   # prevents index out of range error
    for i in range(start + 1, stop):
        if curve[i] > reference_value and curve[i - 1] <= reference_value:
            zlist.append(i - 1)
        elif curve[i] < reference_value and curve[i - 1] >= reference_value:
            zlist.append(i - 1)
    if curve[stop] == reference_value:
        zlist.append(stop)
    return zlist

def residual_analysis(raw, sampling_rate, first_cutoff, last_cutoff, use_critical):
    '''
    Computes the residual between filtered and raw data. Filters from first_cutoff to last_cutoff in steps of 0.5 Hz.
    The function returns a column vector of residuals.
    :param raw: column vector of data to be filtered
    :param sampling_rate: sampling rate in Hz
    :param first_cutoff: 
    :param last_cutoff:
    :param use_critical: boolean set to True to use critical_damped, False for low_pass
    :return: residual vector
    '''
    residual = np.arange(first_cutoff, last_cutoff, 0.5)
    cntr = 0
    for i in np.arange(first_cutoff, last_cutoff, 0.5):
        if use_critical:
            smooth = critically_damped(raw, sampling_rate, i)
        else:
            smooth = low_pass(raw, sampling_rate, i)
        sum = 0.0
        for k in range(len(raw)):
            sum = sum + ((raw[k] - smooth[k]) * (raw[k] - smooth[k]))
        residual[cntr] = np.sqrt(sum/len(raw))
        cntr = cntr + 1
    return residual

def low_pass(raw, sampling_rate, filter_cutoff):
    """
    From the 4th edition of Biomechanics and Motor Control of Human Movement
    by David A. Winter p 69 for filter coefficient corrections.
    This algorithm implements a 4th order zero-phase shift recursive
    Butterworth low pass filter.  Last edited 9-18-2022

    Input parameters
        raw[] is a numpy array containing noise to be removed
        sampling_rate in Hz of the raw[] signal
        filter_cutoff in Hz for the low pass filter

    Output parameters
        smooth[] filtered result
    """
    n = len(raw)
    temp = np.zeros(n + 4, dtype=float)
    prime = np.zeros(n + 4, dtype=float)
    smooth = np.zeros(n, dtype=float)
    sr = sampling_rate
    fc = filter_cutoff
    nPasses = 2.0

    cw = (2.0 ** (1.0 / nPasses) - 1.0) ** (1.0 / 4.0)
    wc = math.tan(math.pi * fc / sr) / cw
    K1 = math.sqrt(2.0) * wc
    K2 = (wc) ** 2.0
    a0 = K2 / (1.0 + K1 + K2)
    a1 = 2.0 * a0
    a2 = a0
    K3 = 2.0 * a0 / K2
    b1 = -2.0 * a0 + K3
    b2 = 1.0 - 2.0 * a0 - K3

    temp[0] = raw[0] + (raw[0] - raw[1])
    temp[1] = raw[0] + (raw[0] - raw[2])
    temp[n + 3] = raw[n - 1] + (raw[n - 1] - raw[n - 2])
    temp[n + 2] = raw[n - 1] + (raw[n - 1] - raw[n - 3])

    for i in range(0, n):
        temp[i + 2] = raw[i]
    for i in range(0, (n + 4)):
        prime[i] = temp[i]
    for i in range(2, (n + 4)):
        prime[i] = a0 * temp[i] + a1 * temp[i - 1] + a2 * \
                   temp[i - 2] + b1 * prime[i - 1] + b2 * prime[i - 2]
    for i in range(0, (n + 4)):
        temp[i] = prime[i]
    for i in range((n + 1), -1, -1):
        prime[i] = a0 * temp[i] + a1 * temp[i + 1] + a2 * \
                   temp[i + 2] + b1 * prime[i + 1] + b2 * prime[i + 2]
    for i in range(0, n):
        smooth[i] = prime[i + 2]
    return smooth


def single_pass(raw, sampling_rate, filter_cutoff):
    """
    From the 4th edition of Biomechanics and Motor Control of Human Movement
    by David A. Winter p 69 for filter coefficient corrections.
    This algorithm implements a 2nd order single pass recursive
    Butterworth low pass filter.  The algorithm will produce a phase shift.
    Last edited 9-18-2022

    Input parameters
        raw[] is a numpy array containing noise to be removed
        sampling_rate in Hz of the raw[] signal
        filter_cutoff in Hz for the low pass filter

    Output parameters
        smooth[] filtered result
    """
    n = len(raw)
    temp = np.zeros(n + 4, dtype=float)
    prime = np.zeros(n + 4, dtype=float)
    smooth = np.zeros(n, dtype=float)
    sr = sampling_rate
    fc = filter_cutoff
    nPasses = 1.0

    cw = (2.0 ** (1.0 / nPasses) - 1.0) ** (1.0 / 4.0)
    wc = math.tan(math.pi * fc / sr) / cw
    K1 = math.sqrt(2.0) * wc
    K2 = (wc) ** 2.0
    a0 = K2 / (1.0 + K1 + K2)
    a1 = 2.0 * a0
    a2 = a0
    K3 = 2.0 * a0 / K2
    b1 = -2.0 * a0 + K3
    b2 = 1.0 - 2.0 * a0 - K3

    temp[0] = raw[0] + (raw[0] - raw[1])
    temp[1] = raw[0] + (raw[0] - raw[2])
    temp[n + 3] = raw[n - 1] + (raw[n - 1] - raw[n - 2])
    temp[n + 2] = raw[n - 1] + (raw[n - 1] - raw[n - 3])

    for i in range(0, n):
        temp[i + 2] = raw[i]
    for i in range(0, (n + 4)):
        prime[i] = temp[i]
    for i in range(2, (n + 4)):
        prime[i] = a0 * temp[i] + a1 * temp[i - 1] + a2 * \
                   temp[i - 2] + b1 * prime[i - 1] + b2 * prime[i - 2]
    for i in range(0, n):
        smooth[i] = prime[i + 2]
    return smooth


def critically_damped(raw, sampling_rate, filter_cutoff):
    """ algorithm implements a 20th order recursive critically damped
        low pass zero-lag Butterworth filter.

        Robertson DG, Dowling JJ (2003) Design and responses of Butterworth and critically
         damped digital filters. J Electromyograph & Kinesiol; 13, 569 - 573.

    Input parameters
        raw[] is a numpy array containing noise to be removed
        sampling_rate in Hz of the raw[] signal
        filter_cutoff in Hz for the low pass filter

    Output parameters
        smooth[] filtered result
    """
    n = len(raw)
    temp = np.zeros(n + 4, dtype=float)
    prime = np.zeros(n + 4, dtype=float)
    smooth = np.zeros(n, dtype=float)
    sr = sampling_rate
    fc = filter_cutoff
    nPasses = 5.0  # five double (forward & backward) passes

    cw = math.sqrt((2.00 ** (1.00 / (2 * nPasses))) - 1.00)
    fc = filter_cutoff
    wc = math.tan(math.pi * fc / sr) / cw
    K1 = 2.00 * wc
    K2 = wc * wc
    a0 = K2 / (1.0 + K1 + K2)
    a1 = 2.00 * a0
    a2 = a0
    K3 = 2.00 * a0 / K2
    b1 = 2.00 * a0 * ((1.0 / K2) - 1.00)
    b2 = 1.00 - (a0 + a1 + a2 + b1)
    # --------------------------------------------------------------
    #                           Pass 1
    # --------------------------------------------------------------
    temp[0] = raw[0] + (raw[0] - raw[1])
    temp[1] = raw[0] + (raw[0] - raw[2])
    temp[n + 3] = raw[n - 1] + (raw[n - 1] - raw[n - 2])
    temp[n + 2] = raw[n - 1] + (raw[n - 1] - raw[n - 3])

    for i in range(0, n):
        temp[i + 2] = raw[i]
    for i in range(0, (n + 4)):
        prime[i] = temp[i]
    for i in range(3, (n + 4)):
        prime[i] = a0 * temp[i] + a1 * temp[i - 1] + a2 * \
                   temp[i - 2] + b1 * prime[i - 1] + b2 * prime[i - 2]
    for i in range(0, (n + 4)):
        temp[i] = prime[i]
    for i in range((n + 1), -1, -1):
        prime[i] = a0 * temp[i] + a1 * temp[i + 1] + a2 * \
                   temp[i + 2] + b1 * prime[i + 1] + b2 * prime[i + 2]
    # --------------------------------------------------------------
    #                           Pass 2
    # --------------------------------------------------------------
    temp[0] = prime[2] + (prime[2] - prime[3])
    temp[1] = prime[2] + (prime[2] - prime[4])
    temp[n + 3] = prime[n + 1] + (prime[n + 1] - prime[n])
    temp[n + 2] = prime[n + 1] + (prime[n + 1] - prime[n - 1])

    for i in range(0, n):
        temp[i + 2] = prime[i + 2]
    for i in range(0, (n + 4)):
        prime[i] = temp[i]
    for i in range(3, (n + 4)):
        prime[i] = a0 * temp[i] + a1 * temp[i - 1] + a2 * \
                   temp[i - 2] + b1 * prime[i - 1] + b2 * prime[i - 2]
    for i in range(0, (n + 4)):
        temp[i] = prime[i]
    for i in range((n + 1), -1, -1):
        prime[i] = a0 * temp[i] + a1 * temp[i + 1] + a2 * \
                   temp[i + 2] + b1 * prime[i + 1] + b2 * prime[i + 2]

    # --------------------------------------------------------------
    #                           Pass 3
    # --------------------------------------------------------------
    temp[0] = prime[2] + (prime[2] - prime[3])
    temp[1] = prime[2] + (prime[2] - prime[4])
    temp[n + 3] = prime[n + 1] + (prime[n + 1] - prime[n])
    temp[n + 2] = prime[n + 1] + (prime[n + 1] - prime[n - 1])

    for i in range(0, n):
        temp[i + 2] = prime[i + 2]
    for i in range(0, (n + 4)):
        prime[i] = temp[i]
    for i in range(3, (n + 4)):
        prime[i] = a0 * temp[i] + a1 * temp[i - 1] + a2 * \
                   temp[i - 2] + b1 * prime[i - 1] + b2 * prime[i - 2]
    for i in range(0, (n + 4)):
        temp[i] = prime[i]
    for i in range((n + 1), -1, -1):
        prime[i] = a0 * temp[i] + a1 * temp[i + 1] + a2 * \
                   temp[i + 2] + b1 * prime[i + 1] + b2 * prime[i + 2]

    # --------------------------------------------------------------
    #                           Pass 4
    # --------------------------------------------------------------
    temp[0] = prime[2] + (prime[2] - prime[3])
    temp[1] = prime[2] + (prime[2] - prime[4])
    temp[n + 3] = prime[n + 1] + (prime[n + 1] - prime[n])
    temp[n + 2] = prime[n + 1] + (prime[n + 1] - prime[n - 1])

    for i in range(0, n):
        temp[i + 2] = prime[i + 2]
    for i in range(0, (n + 4)):
        prime[i] = temp[i]
    for i in range(3, (n + 4)):
        prime[i] = a0 * temp[i] + a1 * temp[i - 1] + a2 * \
                   temp[i - 2] + b1 * prime[i - 1] + b2 * prime[i - 2]
    for i in range(0, (n + 4)):
        temp[i] = prime[i]
    for i in range((n + 1), -1, -1):
        prime[i] = a0 * temp[i] + a1 * temp[i + 1] + a2 * \
                   temp[i + 2] + b1 * prime[i + 1] + b2 * prime[i + 2]

    # --------------------------------------------------------------
    #                           Pass 5
    # --------------------------------------------------------------
    temp[0] = prime[2] + (prime[2] - prime[3])
    temp[1] = prime[2] + (prime[2] - prime[4])
    temp[n + 3] = prime[n + 1] + (prime[n + 1] - prime[n])
    temp[n + 2] = prime[n + 1] + (prime[n + 1] - prime[n - 1])

    for i in range(0, n):
        temp[i + 2] = prime[i + 2]
    for i in range(0, (n + 4)):
        prime[i] = temp[i]
    for i in range(3, (n + 4)):
        prime[i] = a0 * temp[i] + a1 * temp[i - 1] + a2 * \
                   temp[i - 2] + b1 * prime[i - 1] + b2 * prime[i - 2]
    for i in range(0, (n + 4)):
        temp[i] = prime[i]
    for i in range((n + 1), -1, -1):
        prime[i] = a0 * temp[i] + a1 * temp[i + 1] + a2 * \
                   temp[i + 2] + b1 * prime[i + 1] + b2 * prime[i + 2]

    for i in range(0, n):
        smooth[i] = prime[i + 2]
    return smooth  # return the smoothed raw

