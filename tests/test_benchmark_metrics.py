from textvbench.benchmark import aggregate_benchmark_metrics


def test_aggregate_benchmark_metrics_uses_per_sample_means():
    result = aggregate_benchmark_metrics([
        {
            "spatial_fidelity": 0.2,
            "temporal_fidelity": 0.4,
            "textv_score": 20.0,
            "extra_text_ratio": 0.1,
        },
        {
            "spatial_fidelity": 0.8,
            "temporal_fidelity": 0.6,
            "textv_score": 60.0,
            "extra_text_ratio": 0.3,
        },
    ])

    assert result["Spatial"] == result["spatial_fidelity"] == 0.5
    assert result["Temporal"] == result["temporal_fidelity"] == 0.5
    assert result["TextVScore"] == result["textv_score"] == 40.0
    assert result["Extra"] == result["extra_text_ratio"] == 0.2
    assert result["sample_count"] == 2
    assert result["aggregation"] == "arithmetic_mean_over_successful_samples"


def test_aggregate_benchmark_metrics_supports_ref2v_metric_names():
    result = aggregate_benchmark_metrics(
        [{
            "length_weighted_spatial_fidelity": 0.9,
            "length_weighted_temporal_fidelity": 0.8,
            "textv_score": 84.7,
            "extra_text_ratio": 0.05,
        }],
        spatial_key="length_weighted_spatial_fidelity",
        temporal_key="length_weighted_temporal_fidelity",
    )

    assert result["Spatial"] == 0.9
    assert result["Temporal"] == 0.8
    assert result["TextVScore"] == 84.7
    assert result["Extra"] == 0.05
