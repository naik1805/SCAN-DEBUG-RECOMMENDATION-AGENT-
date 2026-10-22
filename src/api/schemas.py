from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class FailureLog(BaseModel):
    mismatch_count: int = Field(..., description="Total number of pattern mismatches observed")
    failing_patterns: List[int] = Field(default_factory=list, description="Indices of failing ATPG patterns")
    failing_cycles: List[int] = Field(default_factory=list, description="Clock cycles at which failures occurred")
    shifter_failure: bool = Field(False, description="True if failure is during shift mode, False if during capture mode")
    defect_type: Optional[str] = Field("NORMAL", description="Wafer defect spatial signature (e.g., CENTER, LOCAL, SCRATCH, etc.)")

class DiagnosisResult(BaseModel):
    suspected_chains: List[str] = Field(default_factory=list, description="Names or IDs of suspected scan chains")
    suspected_cells: List[str] = Field(default_factory=list, description="IDs of suspected failing scan cells/flip-flops")
    fault_models: List[str] = Field(default_factory=list, description="Fault models suspected (e.g., stuck-at, transition, hold, bridge)")

class BitmapCoordinate(BaseModel):
    x: int
    y: int

class FailingBitmap(BaseModel):
    coordinates: List[BitmapCoordinate] = Field(default_factory=list, description="X/Y physical coordinates of failing transistors/cells")
    bounding_box_width: int = Field(0, description="Width of the bounding box of failing coordinates")
    bounding_box_height: int = Field(0, description="Height of the bounding box of failing coordinates")

class HistoricalCase(BaseModel):
    signature_summary: str = Field(..., description="Text description of the failure signature")
    resolved_action: str = Field(..., description="The action that successfully resolved this case")

class ScanAnalysisInput(BaseModel):
    failure_logs: FailureLog
    diagnosis_results: DiagnosisResult
    failing_bitmaps: FailingBitmap
    historical_cases: List[HistoricalCase] = Field(default_factory=list, description="List of relevant historical cases for reference")

class RecommendationResponse(BaseModel):
    recommended_action: str = Field(..., description="The recommended action (one of the 5 action types)")
    sub_recommendations: List[str] = Field(default_factory=list, description="Specific sub-actions for this recommendation category")
    confidence: float = Field(..., description="Model's confidence/Q-value score normalized between 0 and 1")
    rationale: str = Field(..., description="Human-readable text explaining the model's decision")
    detail_metrics: Dict[str, str] = Field(default_factory=dict, description="Detailed category-specific metrics (e.g. Broken Chains, Slack, etc.)")

class FeedbackInput(BaseModel):
    input_data: ScanAnalysisInput
    recommended_action: str
    actual_resolution: str = Field(..., description="The debug action that actually resolved the issue (for reward computation)")
    success: bool = Field(..., description="Whether the recommended action was successful")
