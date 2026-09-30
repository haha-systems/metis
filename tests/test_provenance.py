from metis.provenance import source_fingerprint


def test_source_fingerprint_is_stable_and_sha256_sized():
    first = source_fingerprint()
    assert first == source_fingerprint()
    assert len(first) == 64
