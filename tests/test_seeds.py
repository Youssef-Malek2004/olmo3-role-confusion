from role_confusion.generation import batch_seed


def test_batch_seed_deterministic_and_sensitive():
    a = batch_seed(1, "clean", 1, ["q1", "q2"])
    assert a == batch_seed(1, "clean", 1, ["q1", "q2"])
    assert 0 <= a < 2**31
    assert a != batch_seed(1, "clean", 2, ["q1", "q2"])       # draw changes seed
    assert a != batch_seed(1, "hinted", 1, ["q1", "q2"])      # variant changes seed
    assert a != batch_seed(2, "clean", 1, ["q1", "q2"])       # base seed changes seed
    assert a != batch_seed(1, "clean", 1, ["q2", "q1"])       # composition/order changes seed
