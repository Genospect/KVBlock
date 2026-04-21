"""CLI helpers for the dense-only real-block selector bridge."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

from kvblock.kv.block_modes import VALID_BLOCK_MODES, block_mode_from_name
from kvblock.runtime.base import RuntimeLoadConfig
from kvblock.runtime.hooks import HiddenStateCaptureConfig
from kvblock.runtime.model_loader import create_runtime_backend
from kvblock.kv.qk_aggregation import (
    VALID_QK_AGGREGATION_STRATEGIES,
    qk_aggregation_strategy_from_name,
)
from kvblock.runtime.real_block_eval import (
    BlockInspectionRecord,
    RealBlockSelectorConfig,
    RealBlockSelectorResult,
    run_real_block_selector,
)


@dataclass(frozen=True, slots=True)
class RealBlockSelectorCliResult:
    """CLI execution result plus output-formatting choices."""

    result: RealBlockSelectorResult
    show_selected_blocks: bool
    show_all_blocks: bool
    show_stage_scores: bool
    show_head_diagnostics: bool
    show_query_key_inspection: bool
    show_missed_blocks: bool
    show_top_unselected: int
    json_out_path: Path | None = None
    head_diagnostics_json_out_path: Path | None = None
    inspection_json_out_path: Path | None = None


def build_parser() -> argparse.ArgumentParser:
    """Build the real-block selector CLI parser."""

    parser = argparse.ArgumentParser(
        description="Run dense-only local model ingest through the V1 selector.",
    )
    parser.add_argument(
        "--model",
        default="sshleifer/tiny-gpt2",
        help="Hugging Face causal LM name or local path.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt text. If omitted, --prompt-file is used, then a small default.",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Path to a UTF-8 prompt file.",
    )
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument(
        "--block-mode",
        choices=VALID_BLOCK_MODES,
        default="fixed",
        help=(
            "Candidate block generation mode. Use fixed to honor --block-size, "
            "or named modes like multiscale_16_24_32 for dynamic experiments."
        ),
    )
    parser.add_argument(
        "--overlap-stride",
        type=int,
        default=None,
        help="Optional stride override for overlap block modes.",
    )
    parser.add_argument("--summary-dim", type=int, default=32)
    parser.add_argument("--shortlist-m", type=int, default=16)
    parser.add_argument("--semantic-k", type=int, default=10)
    parser.add_argument("--confidence-margin", type=float, default=0.0)
    parser.add_argument("--keep-recent-blocks", type=int, default=4)
    parser.add_argument("--keep-anchor-blocks", type=int, default=2)
    parser.add_argument(
        "--head-scoring-mode",
        choices=(
            "mean_heads",
            "max_head_score",
            "topk_head_mean",
            "weighted_head_mean",
        ),
        default="mean_heads",
        help="Per-head Stage-A scoring mode for query-key metadata when available.",
    )
    parser.add_argument(
        "--head-top-k",
        type=int,
        default=2,
        help="Head count used by --head-scoring-mode=topk_head_mean.",
    )
    parser.add_argument(
        "--head-weights",
        default="",
        help="Comma-separated static head weights for weighted_head_mean.",
    )
    parser.add_argument(
        "--qk-aggregation",
        choices=VALID_QK_AGGREGATION_STRATEGIES,
        default="mean_pool",
        help="Query/key aggregation strategy for query/key representation sources.",
    )
    parser.add_argument(
        "--top-token-count",
        type=int,
        default=4,
        help="Token count used by --qk-aggregation=top_token_mean.",
    )
    parser.add_argument(
        "--representation-source",
        choices=(
            "final_hidden",
            "hidden_layer_index",
            "middle_hidden",
            "avg_last4_hidden",
            "avg_mid4_hidden",
            "key_mean_last_layer",
            "key_mean_mid_layer",
            "key_avg_last4",
            "query_mean_last_layer",
            "query_mean_mid_layer",
            "query_avg_last4",
        ),
        default="avg_mid4_hidden",
        help="Model-side representation source used for metadata vectors.",
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        default=-1,
        help="Layer index used when --representation-source=hidden_layer_index.",
    )
    parser.add_argument(
        "--show-selected-blocks",
        action="store_true",
        help="Print selected block spans and text previews.",
    )
    parser.add_argument(
        "--show-all-blocks",
        action="store_true",
        help="Print all block spans and text previews.",
    )
    parser.add_argument(
        "--show-stage-scores",
        action="store_true",
        help="Include Stage A/B/final scores in printed block inspection rows.",
    )
    parser.add_argument(
        "--show-head-diagnostics",
        action="store_true",
        help="Print per-head Q/K contribution diagnostics for selected blocks.",
    )
    parser.add_argument(
        "--show-query-key-inspection",
        action="store_true",
        help="Print qualitative query/key block matching groups.",
    )
    parser.add_argument(
        "--show-missed-blocks",
        action="store_true",
        help="Print labeled relevant blocks that were not selected when labels exist.",
    )
    parser.add_argument(
        "--show-top-unselected",
        type=int,
        default=5,
        help="Number of high-scoring unselected blocks to show in query/key inspection.",
    )
    parser.add_argument(
        "--relevance-fragments",
        default="",
        help=(
            "Comma-separated answer/evidence fragments used to label relevant "
            "blocks for qualitative inspection."
        ),
    )
    parser.add_argument(
        "--top-heads",
        type=int,
        default=5,
        help="Number of top contributing heads to keep in diagnostics.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=120,
        help="Maximum decoded text preview characters per block.",
    )
    parser.add_argument(
        "--include-block-text",
        action="store_true",
        help="Include full decoded block text in JSON output.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--torch-dtype",
        default="float32",
        help="Torch dtype for model loading: float32, float16, bfloat16, or auto.",
    )
    parser.add_argument(
        "--device-map",
        default=None,
        help=(
            "Optional Hugging Face device_map for from_pretrained, e.g. auto. "
            "When set, the bridge skips model.to(--device)."
        ),
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require model/tokenizer files to already exist locally.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow HF remote code for local model loading.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional tokenizer truncation length.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path for a structured JSON inspection payload.",
    )
    parser.add_argument(
        "--head-diagnostics-json-out",
        default=None,
        help="Optional path for a structured per-head diagnostic JSON payload.",
    )
    parser.add_argument(
        "--inspection-json-out",
        default=None,
        help="Optional path for a structured query/key qualitative inspection payload.",
    )
    return parser


def run_real_block_selector_cli(
    argv: Sequence[str] | None = None,
) -> RealBlockSelectorCliResult:
    """Run the real-block selector bridge from CLI-style arguments."""

    args = build_parser().parse_args(list(argv) if argv is not None else None)
    prompt_id = _prompt_id_from_args(args)
    prompt_name, inferred_fragments = _prompt_labels_from_args(args)
    explicit_fragments = _parse_relevance_fragments(args.relevance_fragments)
    relevance_fragments = explicit_fragments or inferred_fragments
    emit_query_key_inspection = (
        args.show_query_key_inspection
        or args.show_missed_blocks
        or bool(args.inspection_json_out)
    )
    runtime = create_runtime_backend(
        RuntimeLoadConfig(
            model_name=args.model,
            device=args.device,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
            max_length=args.max_length,
        ),
        capture_config=HiddenStateCaptureConfig(
            representation_source=args.representation_source,
            layer_index=args.layer_index,
        ),
    )
    result = run_real_block_selector(
        runtime,
        read_prompt_from_args(args),
        RealBlockSelectorConfig(
            block_size=args.block_size,
            summary_dim=args.summary_dim,
            shortlist_m=args.shortlist_m,
            semantic_k=args.semantic_k,
            confidence_margin=args.confidence_margin,
            keep_recent_blocks=args.keep_recent_blocks,
            keep_anchor_blocks=args.keep_anchor_blocks,
            head_scoring_mode=args.head_scoring_mode,
            head_top_k=args.head_top_k,
            head_weights=_parse_head_weights(args.head_weights),
            qk_aggregation_strategy=qk_aggregation_strategy_from_name(
                args.qk_aggregation
            ),
            top_token_count=args.top_token_count,
            block_mode=block_mode_from_name(args.block_mode),
            overlap_stride=args.overlap_stride,
            preview_chars=args.preview_chars,
            include_block_text=args.include_block_text or emit_query_key_inspection,
            emit_head_diagnostics=(
                args.show_head_diagnostics or bool(args.head_diagnostics_json_out)
            ),
            top_heads=args.top_heads,
            emit_query_key_inspection=emit_query_key_inspection,
            relevant_text_fragments=relevance_fragments,
            top_unselected_blocks=args.show_top_unselected,
            representation_source=args.representation_source,
            rail_setting="cli",
            prompt_id=prompt_id,
            prompt_name=prompt_name,
        ),
    )
    json_out_path = Path(args.json_out) if args.json_out else None
    if args.json_out:
        json_out_path.parent.mkdir(parents=True, exist_ok=True)
        json_out_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    head_diagnostics_json_out_path = (
        Path(args.head_diagnostics_json_out)
        if args.head_diagnostics_json_out
        else None
    )
    if head_diagnostics_json_out_path is not None:
        head_diagnostics_json_out_path.parent.mkdir(parents=True, exist_ok=True)
        head_diagnostics_json_out_path.write_text(
            json.dumps(_head_diagnostics_payload(result), indent=2),
            encoding="utf-8",
        )
    inspection_json_out_path = (
        Path(args.inspection_json_out) if args.inspection_json_out else None
    )
    if inspection_json_out_path is not None:
        inspection_json_out_path.parent.mkdir(parents=True, exist_ok=True)
        inspection_json_out_path.write_text(
            json.dumps(_query_key_inspection_payload(result), indent=2),
            encoding="utf-8",
        )
    return RealBlockSelectorCliResult(
        result=result,
        show_selected_blocks=args.show_selected_blocks,
        show_all_blocks=args.show_all_blocks,
        show_stage_scores=args.show_stage_scores,
        show_head_diagnostics=args.show_head_diagnostics,
        show_query_key_inspection=(
            args.show_query_key_inspection or args.show_missed_blocks
        ),
        show_missed_blocks=args.show_missed_blocks,
        show_top_unselected=args.show_top_unselected,
        json_out_path=json_out_path,
        head_diagnostics_json_out_path=head_diagnostics_json_out_path,
        inspection_json_out_path=inspection_json_out_path,
    )


def read_prompt_from_args(args: argparse.Namespace) -> str:
    """Resolve prompt text from args using prompt text, file, then default."""

    if args.prompt is not None:
        prompt = args.prompt
    elif args.prompt_file is not None:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    else:
        prompt = "KVBlock builds block metadata from dense model representations."

    if not prompt.strip():
        raise ValueError("prompt must be non-empty")
    return prompt


def format_real_block_selector_summary(
    result: RealBlockSelectorResult,
    *,
    show_selected_blocks: bool = False,
    show_all_blocks: bool = False,
    show_stage_scores: bool = False,
    show_head_diagnostics: bool = False,
    show_query_key_inspection: bool = False,
    show_missed_blocks: bool = False,
    show_top_unselected: int = 5,
) -> str:
    """Format a compact human-readable run summary."""

    confidence = result.confidence
    summary = result.run_summary
    normalized_margin = (
        "n/a" if confidence.normalized_margin is None else f"{confidence.normalized_margin:.6f}"
    )
    normalized_mass = (
        "n/a" if confidence.normalized_mass is None else f"{confidence.normalized_mass:.6f}"
    )
    lines = [
        (
            f"tokens={summary.token_count}, blocks={summary.block_count}, "
            f"block_mode={summary.block_mode}, "
            f"representation={summary.representation_name}"
        ),
        f"selected_block_ids={list(result.selected_block_ids)}",
        (
            f"selected/K={result.selected_to_semantic_k_ratio:.2f}, "
            f"fallback_mode={result.fallback_mode}"
        ),
        (
            f"confidence raw_margin={confidence.raw_margin:.6f}, "
            f"normalized_margin={normalized_margin}, "
            f"selected_mass={confidence.selected_mass}, "
            f"normalized_mass={normalized_mass}, "
            f"is_confident={confidence.is_confident}"
        ),
        (
            f"latency total_sec={result.latency.total_sec:.6f}, "
            f"prefill_sec={result.latency.prefill_sec:.6f}, "
            f"metadata_sec={result.latency.metadata_sec:.6f}, "
            f"selector_sec={result.latency.selector_sec:.6f}, "
            f"inspection_sec={result.latency.inspection_sec:.6f}"
        ),
    ]

    if show_all_blocks:
        lines.append("")
        lines.append("blocks:")
        lines.extend(
            _format_block_inspection(record, show_stage_scores=show_stage_scores)
            for record in result.block_inspections
        )
    elif show_selected_blocks:
        lines.append("")
        lines.append("selected_blocks:")
        lines.extend(
            _format_block_inspection(record, show_stage_scores=show_stage_scores)
            for record in result.selected_block_inspections
        )

    if show_head_diagnostics:
        lines.append("")
        lines.append("head_diagnostics:")
        if not result.head_diagnostics:
            lines.append("unavailable: requires query/key per-head summaries")
        else:
            lines.extend(_format_head_diagnostic(record) for record in result.selected_head_diagnostics)
            if result.head_diagnostic_summary is not None:
                lines.append(
                    "selected_top_heads="
                    + _format_head_frequencies(
                        result.head_diagnostic_summary.selected_top_head_counts
                    )
                )

    if show_query_key_inspection:
        lines.append("")
        lines.append("query_key_inspection:")
        if result.query_key_inspection is None:
            lines.append("unavailable: run with query/key inspection enabled")
        else:
            lines.extend(
                _format_query_key_inspection(
                    result.query_key_inspection,
                    show_missed_blocks=show_missed_blocks,
                    top_unselected=show_top_unselected,
                )
            )

    return "\n".join(lines)


def _format_block_inspection(
    record: BlockInspectionRecord, *, show_stage_scores: bool
) -> str:
    selected = "selected" if record.selected else "unselected"
    line = (
        f"block={record.block_id}, tokens={record.token_start}:"
        f"{record.token_end}, "
        f"candidate={record.candidate_id or 'n/a'}, "
        f"role={record.candidate_role}, "
        f"size={record.block_size or record.token_count}, "
        f"stride={record.stride or record.token_count}, {selected}, "
        f"reason={record.selected_reason}, "
        f"text={record.preview_text!r}"
    )
    if record.parent_candidate_id is not None:
        line += (
            f", parent={record.parent_candidate_id}, "
            f"parent_tokens={record.parent_token_start}:{record.parent_token_end}"
        )
    if show_stage_scores:
        line += (
            f", stage_a={_format_optional_float(record.stage_a_score)}, "
            f"stage_b={_format_optional_float(record.stage_b_score)}, "
            f"final={_format_optional_float(record.final_score)}"
        )
    return line


def _format_optional_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def _format_head_diagnostic(record) -> str:
    top_heads = ", ".join(
        f"h{head.head_index}:{head.score:.3f}"
        for head in record.top_contributing_heads
    )
    return (
        f"block={record.block_id}, tokens={record.token_start}:{record.token_end}, "
        f"reason={record.selected_reason}, aggregate={record.aggregated_score:.6f}, "
        f"top_heads=[{top_heads}], text={record.preview_text!r}"
    )


def _format_head_frequencies(frequencies) -> str:
    if not frequencies:
        return "n/a"
    return ", ".join(
        f"h{item.head_index}:{item.count}({item.fraction:.2f})"
        for item in frequencies
    )


def _head_diagnostics_payload(result: RealBlockSelectorResult) -> dict:
    return {
        "run_summary": result.run_summary.to_dict(),
        "selected_block_ids": list(result.selected_block_ids),
        "head_diagnostic_summary": (
            None
            if result.head_diagnostic_summary is None
            else result.head_diagnostic_summary.to_dict()
        ),
        "head_diagnostics": [
            record.to_dict() for record in result.head_diagnostics
        ],
        "selected_head_diagnostics": [
            record.to_dict() for record in result.selected_head_diagnostics
        ],
    }


def _query_key_inspection_payload(result: RealBlockSelectorResult) -> dict:
    return {
        "run_summary": result.run_summary.to_dict(),
        "selected_block_ids": list(result.selected_block_ids),
        "query_key_inspection": (
            None
            if result.query_key_inspection is None
            else result.query_key_inspection.to_dict()
        ),
    }


def _format_query_key_inspection(
    inspection,
    *,
    show_missed_blocks: bool,
    top_unselected: int,
) -> list[str]:
    groups = inspection.comparison_groups
    lines = [
        (
            f"prompt_id={inspection.prompt_id}, prompt_name={inspection.prompt_name}, "
            f"representation={inspection.representation_source}, "
            f"qk_aggregation={inspection.qk_aggregation_strategy}, "
            f"rail={inspection.rail_setting}"
        ),
        (
            f"selected_relevant={list(groups.selected_relevant_block_ids)}, "
            f"selected_irrelevant={list(groups.selected_irrelevant_block_ids)}, "
            f"missed_relevant={list(groups.missed_relevant_block_ids)}, "
            f"near_miss={list(groups.high_scoring_near_miss_block_ids)}"
        ),
        (
            "query_summary="
            f"dim={inspection.query_summary_metadata.summary_dim}, "
            f"scale={_format_optional_float(inspection.query_summary_metadata.summary_scale)}, "
            f"norm={_format_optional_float(inspection.query_summary_metadata.summary_norm)}, "
            f"heads={inspection.query_summary_metadata.head_count}"
        ),
    ]
    if inspection.relevance_fragments:
        lines.append(f"relevance_fragments={list(inspection.relevance_fragments)}")
    else:
        lines.append("relevance_fragments=none")

    lines.extend(
        _format_query_key_section(
            "selected_relevant_blocks",
            inspection.selected_relevant_blocks,
        )
    )
    lines.extend(
        _format_query_key_section(
            "selected_irrelevant_blocks",
            inspection.selected_irrelevant_blocks,
        )
    )
    if show_missed_blocks:
        lines.extend(
            _format_query_key_section(
                "missed_relevant_blocks",
                inspection.missed_relevant_blocks,
            )
        )
    if top_unselected > 0:
        lines.extend(
            _format_query_key_section(
                "high_scoring_near_miss_blocks",
                inspection.high_scoring_near_miss_blocks[:top_unselected],
            )
        )
    return lines


def _format_query_key_section(name: str, records) -> list[str]:
    lines = [f"{name}:"]
    if not records:
        lines.append("  none")
        return lines
    lines.extend(f"  {_format_query_key_record(record)}" for record in records)
    return lines


def _format_query_key_record(record) -> str:
    relevance = (
        "unlabeled"
        if record.labeled_relevant is None
        else "relevant" if record.labeled_relevant else "not_relevant"
    )
    hints = ", ".join(record.explanation_hints) if record.explanation_hints else "none"
    return (
        f"block={record.block_id}, rank={record.rank_position}, "
        f"tokens={record.token_start}:{record.token_end}, "
        f"candidate={getattr(record, 'candidate_id', None) or 'n/a'}, "
        f"size={getattr(record, 'block_size', None) or record.token_count}, "
        f"reason={record.selected_reason}, {relevance}, "
        f"stage_a={_format_optional_float(record.stage_a_score)}, "
        f"stage_b={_format_optional_float(record.stage_b_score)}, "
        f"final={_format_optional_float(record.final_score)}, "
        f"hints=[{hints}], text={record.preview_text!r}"
    )


def _parse_head_weights(value: str) -> tuple[float, ...]:
    if not value.strip():
        return ()
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _parse_relevance_fragments(value: str) -> tuple[str, ...]:
    if not value.strip():
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _prompt_id_from_args(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).stem
    if args.prompt is not None:
        return "inline_prompt"
    return "default_prompt"


def _prompt_labels_from_args(args: argparse.Namespace) -> tuple[str | None, tuple[str, ...]]:
    if not args.prompt_file:
        return (_prompt_id_from_args(args), ())

    prompt_path = Path(args.prompt_file)
    try:
        resolved_prompt_path = prompt_path.resolve()
    except OSError:
        resolved_prompt_path = prompt_path

    try:
        from kvblock.benchmark.real_block_representation_sweep import (
            default_prompt_retrieval_cases,
        )
    except ImportError:
        return (prompt_path.stem, ())

    for case in default_prompt_retrieval_cases():
        try:
            case_path = case.path.resolve()
        except OSError:
            case_path = case.path
        if resolved_prompt_path == case_path or prompt_path.name == case.path.name:
            return (case.name, case.target_fragments)
    return (prompt_path.stem, ())
