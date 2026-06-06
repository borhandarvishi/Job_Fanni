"""Pydantic schemas for structured OpenAI responses."""

from typing import Literal

from pydantic import BaseModel, Field


class RowJudgment(BaseModel):
    """Judgment for a single job_title ↔ skill pair."""

    id: int = Field(
        description="Numeric id from the input row. Must match the id sent in the prompt."
    )
    is_related: Literal[0, 1] = Field(
        description="1 if the skill is meaningfully related to the job title, 0 otherwise."
    )


class BatchJudgmentResponse(BaseModel):
    """Structured response for a batch of job_title ↔ skill comparisons."""

    results: list[RowJudgment] = Field(
        description="One judgment per input row. Must contain exactly as many items as rows in the batch."
    )
