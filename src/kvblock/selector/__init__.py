"""Heuristic selector skeleton for the V1 scaffold."""

from kvblock.selector.base import FinalSelection, QuerySummary, ScoredBlock
from kvblock.selector.confidence import (
    ConfidenceAssessment,
    ConfidenceEvaluator,
    normalized_selected_mass,
    score_margin,
)
from kvblock.selector.fallback import FallbackDecision, GradedFallbackController
from kvblock.selector.policies import (
    ConfidencePolicy,
    FallbackPolicy,
    StageAPolicy,
    StageAWeights,
    StageBPolicy,
    StageCPolicy,
)
from kvblock.selector.pipeline import (
    SelectorPipeline,
    SelectorPipelineConfig,
    SelectorPipelineResult,
)
from kvblock.selector.oracle import (
    BlockSetComparison,
    DenseReferenceBlock,
    DenseReferenceBlockSet,
    SparseSelectedBlockSet,
    SyntheticDenseOracle,
    SyntheticDenseOracleConfig,
    compare_block_sets,
    dense_reference_block_set,
    sparse_selected_block_set,
)
from kvblock.selector.stage_a import StageAScorer, approx_cosine_similarity
from kvblock.selector.stage_b import StageBRefiner
from kvblock.selector.stage_c import StageCSelector
from kvblock.selector.trace import (
    BlockScoreTrace,
    ConfidenceTrace,
    FallbackTrace,
    SelectionSplitTrace,
    SelectorDecisionTrace,
)

__all__ = [
    "ConfidenceAssessment",
    "ConfidenceEvaluator",
    "ConfidencePolicy",
    "ConfidenceTrace",
    "BlockScoreTrace",
    "BlockSetComparison",
    "DenseReferenceBlock",
    "DenseReferenceBlockSet",
    "FallbackDecision",
    "FallbackPolicy",
    "FallbackTrace",
    "FinalSelection",
    "GradedFallbackController",
    "QuerySummary",
    "ScoredBlock",
    "SelectionSplitTrace",
    "StageAPolicy",
    "StageAScorer",
    "StageAWeights",
    "StageBPolicy",
    "StageBRefiner",
    "StageCPolicy",
    "StageCSelector",
    "SparseSelectedBlockSet",
    "SelectorDecisionTrace",
    "SelectorPipeline",
    "SelectorPipelineConfig",
    "SelectorPipelineResult",
    "SyntheticDenseOracle",
    "SyntheticDenseOracleConfig",
    "approx_cosine_similarity",
    "compare_block_sets",
    "dense_reference_block_set",
    "normalized_selected_mass",
    "score_margin",
    "sparse_selected_block_set",
]
