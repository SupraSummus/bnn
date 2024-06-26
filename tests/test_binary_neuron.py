import numpy

from bnn import BN


def test_empty():
    bn = BN(sources_dim=0, sources=numpy.array([]))

    signal = numpy.zeros(shape=(0,), dtype=bool)
    bn.infer(numpy.array([]), signal)

    too_much = numpy.zeros(shape=(0,), dtype=float)
    not_enough = numpy.zeros(shape=(0,), dtype=float)
    bn.compute_error(numpy.array([]), signal, too_much, not_enough)


def test_inference():
    bn = BN(
        sources_dim=2,
        sources=numpy.array([
            (-1, 0),
            (-1, -2),
            (-2, 1),
        ]),
    )
    bn.excitators = [0, 0, 1, 2]
    bn.activation_treshold = 2
    signal = numpy.zeros(shape=(5, 5, 6), dtype=bool)
    signal[2, 2, 0] = True
    signal[2, 0, 1] = True
    signal[3, 2, 1] = True
    signal[1, 3, 2] = True
    signal[(2, 1), (0, 3), 3] = True
    orig_signal = signal.copy()
    bn.infer(numpy.array([(3, 2), (4, 2)]), signal)

    # our inference is correct
    assert list(signal[3, 2]) == [True, False, False, True, False, False]
    assert list(signal[4, 2]) == [False, True, False, False, False, False]

    # we havent modified anything else
    signal[3, 2] = orig_signal[3, 2]
    signal[4, 2] = orig_signal[4, 2]
    assert (signal == orig_signal).all()


def test_compute_error():
    bn = BN(
        sources_dim=1,
        sources=numpy.array([
            (-3,),
            (-2,),
            (-1,),
        ]),
    )
    bn.excitators = [0, 2]
    bn.inhibitors = [1]
    signal = numpy.zeros(shape=(5, 6), dtype=bool)
    too_much = numpy.zeros(shape=(5, 6), dtype=float)
    not_enough = numpy.zeros(shape=(5, 6), dtype=float)

    too_much[3, 0] = 0.5

    not_enough[3, 1] = 0.3

    too_much[3, 2] = 0.4
    not_enough[3, 2] = 0.6

    signal[(0, 1), 3] = True
    too_much[3, 3] = 0.4
    not_enough[3, 3] = 0.6

    signal[(0, 1, 2, 3, 4), 4] = True
    too_much[3, 4] = 0.4
    too_much[4, 4] = 0.5
    not_enough[3, 4] = 0.6
    not_enough[4, 4] = 0.7

    bn.compute_error(numpy.array([(3,), (4,)]), signal, too_much, not_enough)

    assert numpy.allclose(too_much, [
        [.0, .0,  .0, .4, .2,        .0],  # noqa: E241
        [.0, .0,  .0, .3, .25 + 0.6, .0],  # noqa: E241
        [.0, .0,  .0, .0, .2 + 0.7,  .0],  # noqa: E241
        [.5, .0,  .4, .4, .4 + 0.25, .0],  # noqa: E241
        [.0, .0,  .0, .0, .5,        .0],  # noqa: E241
    ])
    assert numpy.allclose(not_enough, [
        [.0, .15, .3, .0, .0,        .0],  # noqa: E241
        [.5, .0,  .4, .0, .0,        .0],  # noqa: E241
        [.0, .15, .3, .3, .0,        .0],  # noqa: E241
        [.0, .3,  .6, .6, .6,        .0],  # noqa: E241
        [.0, .0,  .0, .0, .7,        .0],  # noqa: E241
    ])


def test_comupte_error_discrete():
    bn = BN(
        sources_dim=1,
        sources=numpy.array([
            (-4,),
            (-3,),
            (-2,),
            (-1,),
        ]),
    )
    bn.excitators = [0, 1]
    bn.inhibitors = [2, 3]
    signal = numpy.zeros(shape=(5, 8), dtype=bool)
    too_much = numpy.zeros(shape=(5, 8), dtype=numpy.uint8)
    not_enough = numpy.zeros(shape=(5, 8), dtype=numpy.uint8)

    too_much[4, (0, 1, 2, 3)] = 128
    not_enough[4, (0, 1, 4, 5)] = 121
    signal[0, (0, 2, 4, 6)] = True
    signal[2, (0, 2, 4, 6)] = True
    signal[1] = True

    bn.compute_error(numpy.array([(4,)]), signal, too_much, not_enough, dtype=numpy.uint8)

    assert numpy.equal(too_much, [
        [42,  0,   42,  0,   0,   0,   0, 0],  # noqa: E241
        [42,  42,  42,  42,  0,   0,   0, 0],  # noqa: E241
        [121, 0,   0,   0,   121, 0,   0, 0],  # noqa: E241
        [0,   0,   0,   0,   0,   0,   0, 0],  # noqa: E241
        [128, 128, 128, 128, 0,   0,   0, 0],  # noqa: E241
    ]).all()
    assert numpy.equal(not_enough, [
        [0,   121, 0,   0,   0,   121, 0, 0],  # noqa: E241
        [0,   0,   0,   0,   0,   0,   0, 0],  # noqa: E241
        [0,   42,  0,   42,  0,   0,   0, 0],  # noqa: E241
        [42,  42,  42,  42,  0,   0,   0, 0],  # noqa: E241
        [121, 121, 0,   0,   121, 121, 0, 0],  # noqa: E241
    ]).all()


def test_signal_error_correlation():
    bn = BN(
        sources_dim=1,
        sources=numpy.array([
            (-4,),
            (-3,),
            (-2,),
            (-1,),
        ]),
    )
    bn.excitators = [0, 1, 1]
    bn.inhibitors = [2, 3, 1]
    signal = numpy.zeros(shape=(10, 2), dtype=bool)
    too_much = numpy.zeros(shape=(10, 2), dtype=numpy.uint8)
    not_enough = numpy.zeros(shape=(10, 2), dtype=numpy.uint8)

    signal[(0, 1, 5, 7), 0] = True
    too_much[4, 0] = 54
    too_much[9, 0] = 5

    signal[:, 1] = True
    too_much[:, 1] = 2
    not_enough[:, 1] = 1

    too_much_correlation, not_enough_correlation = bn.get_signal_error_correlation(
        numpy.array([
            (4,),
            (9,),
        ]),
        signal, too_much, not_enough,
        dtype=numpy.uint8,
    )

    assert numpy.equal(too_much_correlation, [
        [59, 4],
        [54, 4],
        [5, 4],
        [0, 4],
    ]).all()
    assert numpy.equal(not_enough_correlation, [
        [0, 2],
        [0, 2],
        [0, 2],
        [0, 2],
    ]).all()
