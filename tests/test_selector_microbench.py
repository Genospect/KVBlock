from kvblock.benchmark.selector_microbench import (
    SelectorMicrobenchSpec,
    generate_synthetic_selector_population,
    run_selector_microbench_case,
    run_selector_microbench_sweep,
)


def test_selector_microbench_generation_is_deterministic_with_seed() -> None:
    spec = SelectorMicrobenchSpec(
        case_id="deterministic",
        num_blocks=8,
        num_queries=3,
        seed=123,
    )

    first = generate_synthetic_selector_population(spec)
    second = generate_synthetic_selector_population(spec)

    assert [item.to_dict() for item in first.metadata_blocks] == [
        item.to_dict() for item in second.metadata_blocks
    ]
    assert [query.tolist() for query in first.query_summaries] == [
        query.tolist() for query in second.query_summaries
    ]


def test_selector_microbench_sweep_executes_multiple_block_counts() -> None:
    specs = [
        SelectorMicrobenchSpec(case_id="small", num_blocks=8, num_queries=2, seed=1),
        SelectorMicrobenchSpec(case_id="large", num_blocks=16, num_queries=2, seed=2),
    ]

    results = run_selector_microbench_sweep(specs)

    assert [result.spec.num_blocks for result in results] == [8, 16]
    assert [result.query_count for result in results] == [2, 2]


def test_selector_microbench_metrics_output_contains_required_fields() -> None:
    spec = SelectorMicrobenchSpec(case_id="fields", num_blocks=8, num_queries=1, seed=3)

    result = run_selector_microbench_case(spec)
    row = result.rows[0].to_dict()

    required = {
        "selector_latency_sec",
        "stage_a_candidate_count",
        "stage_a_shortlist_size",
        "stage_b_refinement_count",
        "final_selected_block_count",
        "semantic_selected_block_count",
        "rail_preserved_block_count",
        "fallback_mode",
        "raw_margin",
        "normalized_margin",
        "selected_mass",
        "normalized_mass",
        "trace_size_bytes",
        "oracle_recall_rate",
        "oracle_precision_rate",
        "oracle_overlap_count",
    }

    assert required.issubset(row.keys())


def test_selector_microbench_oracle_disabled_run_still_works() -> None:
    spec = SelectorMicrobenchSpec(
        case_id="oracle-off",
        num_blocks=8,
        num_queries=2,
        seed=7,
        oracle_enabled=False,
    )

    result = run_selector_microbench_case(spec)

    assert result.oracle_summary is None
    assert all(row.oracle_recall_rate is None for row in result.rows)


def test_selector_microbench_oracle_enabled_run_emits_quality_metrics() -> None:
    spec = SelectorMicrobenchSpec(
        case_id="oracle-on",
        num_blocks=8,
        num_queries=2,
        seed=8,
        oracle_enabled=True,
        oracle_top_k=2,
    )

    result = run_selector_microbench_case(spec)
    row = result.rows[0]

    assert row.oracle_recall_rate is not None
    assert row.oracle_precision_rate is not None
    assert row.oracle_overlap_count is not None
    assert row.oracle_missed_important_count is not None
    assert row.oracle_extra_selected_count is not None


def test_selector_microbench_aggregate_metrics_include_oracle_statistics() -> None:
    spec = SelectorMicrobenchSpec(
        case_id="oracle-agg",
        num_blocks=10,
        num_queries=3,
        seed=9,
        oracle_enabled=True,
    )

    result = run_selector_microbench_case(spec)

    assert result.oracle_summary is not None
    assert result.oracle_summary.mean_recall_rate >= 0.0
    assert result.oracle_summary.mean_precision_rate >= 0.0
    assert result.oracle_summary.mean_overlap_count >= 0.0


def test_selector_microbench_tracks_fallback_frequency() -> None:
    spec = SelectorMicrobenchSpec(
        case_id="fallbacks",
        num_blocks=12,
        num_queries=4,
        seed=4,
        population_profile="low_confidence",
        semantic_top_k=1,
        confidence_margin=0.2,
        widen_top_k_by=1,
    )

    result = run_selector_microbench_case(spec)

    assert sum(result.fallback_frequency_by_mode.values()) == spec.num_queries
    assert any(mode in result.fallback_frequency_by_mode for mode in {"widen_k", "dense", "sparse", "add_recent"})


def test_selector_microbench_oracle_behavior_is_deterministic_with_fixed_seed() -> None:
    spec = SelectorMicrobenchSpec(
        case_id="oracle-deterministic",
        num_blocks=8,
        num_queries=3,
        seed=10,
        oracle_enabled=True,
    )

    first = run_selector_microbench_case(spec)
    second = run_selector_microbench_case(spec)

    def comparable_rows(result):
        rows = []
        for row in result.rows:
            payload = row.to_dict()
            payload.pop("selector_latency_sec")
            rows.append(payload)
        return rows

    assert comparable_rows(first) == comparable_rows(second)
    assert (
        None if first.oracle_summary is None else first.oracle_summary.to_dict()
    ) == (
        None if second.oracle_summary is None else second.oracle_summary.to_dict()
    )


def test_selector_microbench_handles_rail_dominated_and_low_confidence_cases() -> None:
    specs = [
        SelectorMicrobenchSpec(
            case_id="rails",
            num_blocks=10,
            num_queries=2,
            seed=5,
            population_profile="rail_dominated",
            keep_recent_blocks=3,
            keep_anchor_blocks=2,
            oracle_enabled=True,
        ),
        SelectorMicrobenchSpec(
            case_id="lowconf",
            num_blocks=10,
            num_queries=2,
            seed=6,
            population_profile="low_confidence",
            confidence_margin=0.2,
            semantic_top_k=1,
            oracle_enabled=True,
        ),
    ]

    results = run_selector_microbench_sweep(specs)

    assert len(results) == 2
    assert all(result.query_count == 2 for result in results)
    assert all(result.rows for result in results)
    assert all(result.oracle_summary is not None for result in results)
    assert all(row.oracle_recall_rate is not None for result in results for row in result.rows)


def test_selector_microbench_looser_confidence_margin_reduces_widening() -> None:
    strict = run_selector_microbench_case(
        SelectorMicrobenchSpec(
            case_id="strict",
            num_blocks=16,
            num_queries=3,
            seed=17,
            confidence_margin=0.2,
        )
    )
    loose = run_selector_microbench_case(
        SelectorMicrobenchSpec(
            case_id="loose",
            num_blocks=16,
            num_queries=3,
            seed=17,
            confidence_margin=0.0,
        )
    )

    assert strict.fallback_frequency_by_mode.get("widen_k", 0) > 0
    assert loose.fallback_frequency_by_mode.get("widen_k", 0) == 0
    assert loose.fallback_frequency_by_mode.get("sparse", 0) == loose.query_count


def test_selector_microbench_sweep_is_stable_for_calibration_cases() -> None:
    specs = [
        SelectorMicrobenchSpec(
            case_id="m12-k4-c0",
            num_blocks=12,
            shortlist_size=12,
            semantic_top_k=4,
            confidence_margin=0.0,
            num_queries=2,
            seed=18,
            oracle_enabled=True,
        ),
        SelectorMicrobenchSpec(
            case_id="m8-k2-c20",
            num_blocks=12,
            shortlist_size=8,
            semantic_top_k=2,
            confidence_margin=0.2,
            num_queries=2,
            seed=19,
            oracle_enabled=True,
        ),
    ]

    results = run_selector_microbench_sweep(specs)

    assert [result.spec.shortlist_size for result in results] == [12, 8]
    assert [result.spec.semantic_top_k for result in results] == [4, 2]
    assert [result.spec.confidence_margin for result in results] == [0.0, 0.2]
    assert all(result.query_count == 2 for result in results)
