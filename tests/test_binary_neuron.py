from bnn import BN
import numpy


def test_inference_empty():
    bn = BN(sources_dim=0, sources=numpy.array([]))
    result = numpy.zeros(shape=(0,), dtype=bool)
    bn.infer((), result)
    assert result.shape == (0,)


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
    result = numpy.zeros(shape=(5, 5, 6), dtype=bool)
    result[2, 2, 0] = True
    result[2, 0, 1] = True
    result[1, 3, 2] = True
    result[(2, 1), (0, 3), 3] = True
    orig = result.copy()
    bn.infer((3, 2), result)

    # our inference is correct
    assert list(result[3, 2]) == [True, False, False, True, False, False]

    # we havent modified anything else
    result[3, 2] = False
    assert (result == orig).all()

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
    result = numpy.zeros(shape=(4, 6), dtype=bool)
    too_much = numpy.zeros(shape=(4, 6), dtype=float)
    not_enough = numpy.zeros(shape=(4, 6), dtype=float)

    too_much[3, 0] = 0.5

    not_enough[3, 1] = 0.3

    too_much[3, 2] = 0.4
    not_enough[3, 2] = 0.6

    result[(0, 1), 3] = True
    too_much[3, 3] = 0.4
    not_enough[3, 3] = 0.6

    bn.compute_error((3,), result, too_much, not_enough)

    assert numpy.allclose(too_much[:3], [
        [.0, .0, .0, .4, .0, .0],
        [.0, .0, .0, .3, .0, .0],
        [.0, .0, .0, .0, .0, .0],
    ])
    assert numpy.allclose(not_enough[:3], [
        [.0, .15, .3, .0, .0, .0],
        [.5, .0, .4, .0, .0, .0],
        [.0, .15, .3, .3, .0, .0],
    ])
