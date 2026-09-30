from training.synthetic import generate_dataset


def test_synthetic_generation_is_seed_deterministic():
    first = generate_dataset(40, seed=41)
    second = generate_dataset(40, seed=41)
    assert first == second

