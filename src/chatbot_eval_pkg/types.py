# src/chatbot_eval_pkg/types.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class MetricResultDetail:
    """Detailed results for a single metric."""
    metric: str
    score: Optional[float] = None
    threshold: Optional[float] = None
    success: bool = False
    reason: Optional[str] = None
    error: Optional[str] = None

@dataclass
class EvaluationResult:
    """Structured result for a single test case evaluation."""
    id: str
    success: bool
    duration: float
    chatbot_response: Optional[str] = None
    retrieval_context_extracted: Optional[List[str]] = None
    metrics_results: List[MetricResultDetail] = field(default_factory=list)
    error: Optional[str] = None
    file_path: Optional[str] = None # Original file path for reference
    test_case_details: Optional[Dict[str, Any]] = None # Original test case data
