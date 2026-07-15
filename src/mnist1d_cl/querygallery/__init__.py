"""Query-gallery representation-compatibility evaluation."""

from .retrieval import (
    l2_normalize,
    cosine_sim_matrix,
    query_gallery_eval,
    retrieval_error_analysis,
    audit_query_gallery,
    audit_hundred_percent,
)

__all__ = [
    "l2_normalize",
    "cosine_sim_matrix",
    "query_gallery_eval",
    "retrieval_error_analysis",
    "audit_query_gallery",
    "audit_hundred_percent",
]
