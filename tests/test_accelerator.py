from copperfern.accelerator import accumulate_gradients


def test_accumulate_windows():
    assert accumulate_gradients([1.0, 2.0, 3.0, 4.0], 2) == [1.5, 3.5]
