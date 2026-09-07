import pytest

from mds_norm.pipeline.institutional_priors import eq_rangelike, prime_evidence


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # flag-affected, ascending: genuine positive evidence
        ("4.1914=3.1915", True),
        ("1930=1932", True),
        ("[7.1883=12.1883]?", True),  # brackets/qualifier handled by the parser
        # flag-affected, descending: genuine negative evidence
        ("12.1883=7.1883", False),
        ("1932=1930", False),
        # unparseable under both conventions, so no evidence
        ("217BC=215BC", None),
        ("=31.3.1969", None),
        ("1938=", None),
        ("<=1974", None),
        ("1923=1924<onwards", None),
        ("[11=12.1887]", None),
    ],
)
def test_eq_rangelike(value, expected):
    assert eq_rangelike(value) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2.98' (75.7mm)", "in"),  # 2.98 in = 75.7 mm — the Royal Armouries case
        ("3' (914mm)", "ft"),  # 3 ft = 914.4 mm
        ("5' (152.4cm)", "ft"),
        ("blade 30.5' (77.5cm)", "in"),
        ("no primes here", None),  # no dual annotation
        ("3' (100mm)", None),  # fits neither reading
        ("2.98' (75.7mm) and 3' (914mm)", None),  # contradictory within one value
    ],
)
def test_prime_evidence(value, expected):
    assert prime_evidence(value) == expected
