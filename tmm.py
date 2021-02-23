import numpy as np


def transmission(n, wavelengths, d):
    n_sum = n[:-1] + n[1:]
    r = (n[1:] - n[:-1]) / n_sum
    t = 2 * n[:-1] / n_sum
    delta = d * 2 * np.pi * n[1:-1] / wavelengths

    return np.power(np.abs(t[0] * t[1] / (np.exp(-1j * delta[0]) + r[0] * r[1] * np.exp(1j * delta[0]))), 2) * n[-1].real


if __name__ == "__main__":
    test_args = {
        'n': np.array([[1, 1],
                       [2.3257965013488495 + 0.17743239666374633j, 2.4174928867085916 + 0.01811159390480943j],
                       [1.5, 1.5]]),
        'wavelengths': np.array([566.4599614767255, 666.5024975922954]),
        'd': 540
    }
    trans = transmission(test_args['n'], test_args['wavelengths'], test_args['d'])
    print(trans)
