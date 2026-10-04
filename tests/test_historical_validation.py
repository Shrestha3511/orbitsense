from historical_validation import compute_historical_metrics


def test_historical_metrics_are_reasonable_without_network():
    metrics = compute_historical_metrics(search_seconds=180, step_seconds=1.0)
    assert 0 < metrics["miss_m"] < 5000
    assert 10.0 < metrics["rel_v"] < 13.0
