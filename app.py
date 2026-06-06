"""Classify job_title ↔ skill relations using OpenAI GPT with checkpoint resume."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.responses import Response
from openai.types.responses.response_usage import ResponseUsage

from prompt import SYSTEM_PROMPT, build_user_prompt
from schema import BatchJudgmentResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_CSV = Path("job_skill.csv")
OUTPUT_CSV = Path("job_skill_output.csv")
CHECKPOINT_FILE = Path("checkpoint.json")
USAGE_CSV = Path("request_usage.csv")

BATCH_SIZE = 100
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4")
MAX_RETRIES = 5
RETRY_BASE_DELAY_SEC = 2.0
DEFAULT_MAX_WORKERS = max(4, (os.cpu_count() or 4) * 2)

_state_lock = threading.Lock()
_print_lock = threading.Lock()

# GPT-5.4 standard pricing (USD per 1M tokens) — input <= 272K context
PRICE_INPUT_PER_M = 2.50
PRICE_CACHED_INPUT_PER_M = 0.25
PRICE_OUTPUT_PER_M = 15.00

# GPT-5.4 long-context pricing — input > 272K context
LONG_CONTEXT_THRESHOLD = 272_000
PRICE_INPUT_LONG_PER_M = 5.00
PRICE_CACHED_INPUT_LONG_PER_M = 0.50
PRICE_OUTPUT_LONG_PER_M = 22.50

USAGE_COLUMNS = ["id", "created_at", "input_token", "output_token", "price", "model"]


@dataclass(frozen=True)
class BatchResult:
    """Parsed judgments plus API usage metadata for one request."""

    judgments: BatchJudgmentResponse
    usage_records: list[dict[str, str | int | float]]


@dataclass(frozen=True)
class BatchJob:
    """One pending batch task."""

    batch_num: int
    total_batches: int
    rows: list[dict]


def safe_print(message: str) -> None:
    """Thread-safe console output."""
    with _print_lock:
        print(message, flush=True)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def load_checkpoint() -> set[int]:
    """Return the set of job_skill ids already processed."""
    if not CHECKPOINT_FILE.exists():
        return set()
    with CHECKPOINT_FILE.open(encoding="utf-8") as f:
        data = json.load(f)
    # Support legacy checkpoint key for older runs.
    ids = data.get("processed_ids", data.get("processed_indices", []))
    return {int(value) for value in ids}


def save_checkpoint(processed_ids: set[int]) -> None:
    """Persist processed job_skill ids to disk."""
    with CHECKPOINT_FILE.open("w", encoding="utf-8") as f:
        json.dump(
            {"processed_ids": sorted(processed_ids)},
            f,
            ensure_ascii=False,
            indent=2,
        )


def load_or_init_output(df: pd.DataFrame) -> pd.DataFrame:
    """Load existing output CSV or create a fresh output frame."""
    if OUTPUT_CSV.exists():
        output_df = pd.read_csv(OUTPUT_CSV)
        if "is_related" not in output_df.columns:
            output_df["is_related"] = pd.NA
        return output_df

    output_df = df.copy()
    output_df["is_related"] = pd.NA
    return output_df


def save_output(output_df: pd.DataFrame) -> None:
    """Write the output CSV."""
    output_df.to_csv(OUTPUT_CSV, index=False)


# ---------------------------------------------------------------------------
# Usage / pricing helpers
# ---------------------------------------------------------------------------


def calculate_price_usd(usage: ResponseUsage) -> float:
    """Calculate request cost in USD from token usage (GPT-5.4 pricing)."""
    cached_tokens = usage.input_tokens_details.cached_tokens
    non_cached_input = usage.input_tokens - cached_tokens

    if usage.input_tokens > LONG_CONTEXT_THRESHOLD:
        input_rate = PRICE_INPUT_LONG_PER_M / 1_000_000
        cached_rate = PRICE_CACHED_INPUT_LONG_PER_M / 1_000_000
        output_rate = PRICE_OUTPUT_LONG_PER_M / 1_000_000
    else:
        input_rate = PRICE_INPUT_PER_M / 1_000_000
        cached_rate = PRICE_CACHED_INPUT_PER_M / 1_000_000
        output_rate = PRICE_OUTPUT_PER_M / 1_000_000

    return (
        non_cached_input * input_rate
        + cached_tokens * cached_rate
        + usage.output_tokens * output_rate
    )


def split_usage_across_ids(
    response: Response,
    batch_ids: list[int],
) -> list[dict[str, str | int | float]]:
    """Split one API request's usage evenly across job_skill ids in the batch."""
    if response.usage is None:
        raise RuntimeError(f"Response {response.id} has no usage data.")
    if not batch_ids:
        raise RuntimeError("Cannot build usage records for an empty batch.")

    created_at = datetime.fromtimestamp(
        response.created_at,
        tz=timezone.utc,
    ).isoformat()

    total_input = response.usage.input_tokens
    total_output = response.usage.output_tokens
    total_price = calculate_price_usd(response.usage)
    model = str(response.model)
    n = len(batch_ids)

    base_input = total_input // n
    base_output = total_output // n
    base_price = total_price / n

    input_remainder = total_input - base_input * n
    output_remainder = total_output - base_output * n
    price_remainder = total_price - base_price * n

    records: list[dict[str, str | int | float]] = []
    for i, row_id in enumerate(batch_ids):
        input_token = base_input + (1 if i < input_remainder else 0)
        output_token = base_output + (1 if i < output_remainder else 0)
        price = base_price + (price_remainder if i == n - 1 else 0.0)

        records.append(
            {
                "id": int(row_id),
                "created_at": created_at,
                "input_token": input_token,
                "output_token": output_token,
                "price": round(price, 8),
                "model": model,
            }
        )

    return records


def append_usage_records(records: list[dict[str, str | int | float]]) -> None:
    """Append request usage rows to the usage CSV (caller must hold _state_lock)."""
    if not records:
        return
    row_df = pd.DataFrame(records, columns=USAGE_COLUMNS)
    write_header = not USAGE_CSV.exists()
    row_df.to_csv(USAGE_CSV, mode="a", header=write_header, index=False)


def load_total_spent_usd() -> float:
    """Return total USD spent from existing usage CSV."""
    if not USAGE_CSV.exists():
        return 0.0
    usage_df = pd.read_csv(USAGE_CSV)
    if usage_df.empty or "price" not in usage_df.columns:
        return 0.0
    return float(usage_df["price"].sum())


# ---------------------------------------------------------------------------
# OpenAI batch classification
# ---------------------------------------------------------------------------


def classify_batch(
    client: OpenAI,
    rows: list[dict],
) -> BatchResult:
    """Send one batch to the model and return structured judgments plus usage."""
    response = client.responses.parse(
        model=MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(rows)},
        ],
        text_format=BatchJudgmentResponse,
    )

    if hasattr(response, "output") and response.output:
        first = response.output[0]
        if getattr(first, "type", None) == "refusal":
            raise RuntimeError(f"Model refused the request: {first}")

    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError("Model returned no parsed structured output.")

    if len(parsed.results) != len(rows):
        raise RuntimeError(
            f"Expected {len(rows)} results, got {len(parsed.results)}."
        )

    expected_ids = {row["id"] for row in rows}
    returned_ids = {r.id for r in parsed.results}
    if expected_ids != returned_ids:
        missing = expected_ids - returned_ids
        extra = returned_ids - expected_ids
        raise RuntimeError(
            f"Id mismatch. Missing: {sorted(missing)}. Extra: {sorted(extra)}."
        )

    batch_ids = [row["id"] for row in rows]
    return BatchResult(
        judgments=parsed,
        usage_records=split_usage_across_ids(response, batch_ids),
    )


def call_with_retry(
    client: OpenAI,
    rows: list[dict],
    batch_label: str = "",
) -> BatchResult:
    """Call classify_batch with exponential backoff on transient errors."""
    last_error: Exception | None = None
    prefix = f"[{batch_label}] " if batch_label else ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return classify_batch(client, rows)
        except Exception as exc:  # noqa: BLE001 — retry on any API/network failure
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            delay = RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1))
            safe_print(
                f"{prefix}Attempt {attempt}/{MAX_RETRIES} failed: {exc}. "
                f"Retrying in {delay:.0f}s..."
            )
            time.sleep(delay)

    raise RuntimeError(f"Batch failed after {MAX_RETRIES} attempts.") from last_error


def process_batch_job(api_key: str, job: BatchJob) -> tuple[BatchJob, BatchResult]:
    """Run one batch in a worker thread."""
    client = OpenAI(api_key=api_key)
    batch_ids = [row["id"] for row in job.rows]
    batch_label = (
        f"batch {job.batch_num}/{job.total_batches} "
        f"ids {batch_ids[0]}..{batch_ids[-1]}"
    )
    safe_print(f"Started {batch_label} ({len(job.rows)} rows)")
    result = call_with_retry(client, job.rows, batch_label=batch_label)
    return job, result


def persist_batch_result(
    output_df: pd.DataFrame,
    processed_ids: set[int],
    batch_result: BatchResult,
    total_rows: int,
    job: BatchJob,
) -> float:
    """Apply one batch result and persist checkpoint/output/usage."""
    batch_ids = [row["id"] for row in job.rows]

    with _state_lock:
        apply_batch_results(output_df, batch_result.judgments, processed_ids)
        append_usage_records(batch_result.usage_records)
        save_output(output_df)
        save_checkpoint(processed_ids)

        batch_cost = sum(
            float(record["price"]) for record in batch_result.usage_records
        )
        related_count = sum(
            r.is_related == 1 for r in batch_result.judgments.results
        )
        total_input = sum(
            record["input_token"] for record in batch_result.usage_records
        )
        total_output = sum(
            record["output_token"] for record in batch_result.usage_records
        )
        processed_count = len(processed_ids)

    safe_print(
        f"Finished batch {job.batch_num}/{job.total_batches} "
        f"— ids {batch_ids[0]}..{batch_ids[-1]} | "
        f"related={related_count}/{len(job.rows)} | "
        f"processed={processed_count}/{total_rows} | "
        f"tokens in={total_input} out={total_output} | "
        f"cost=${batch_cost:.8f}"
    )
    return batch_cost


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def validate_input(df: pd.DataFrame) -> None:
    """Ensure the input CSV has the required columns."""
    required = {"id", "job_title", "skill"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV must contain columns {sorted(required)}. "
            f"Missing: {sorted(missing)}"
        )

    if df["id"].isna().any():
        raise ValueError("Input CSV contains rows with missing id values.")

    if df["id"].duplicated().any():
        duplicates = df.loc[df["id"].duplicated(), "id"].tolist()
        raise ValueError(f"Input CSV contains duplicate id values: {duplicates}")


def build_pending_batches(
    df: pd.DataFrame,
    processed_ids: set[int],
    batch_size: int,
) -> list[list[dict]]:
    """Group unprocessed rows into batches."""
    pending_rows: list[dict] = []
    for _, row in df.iterrows():
        row_id = int(row["id"])
        if row_id in processed_ids:
            continue
        pending_rows.append(
            {
                "id": row_id,
                "job_title": str(row["job_title"]).strip(),
                "skill": str(row["skill"]).strip(),
            }
        )

    batches: list[list[dict]] = []
    for start in range(0, len(pending_rows), batch_size):
        batches.append(pending_rows[start : start + batch_size])
    return batches


def apply_batch_results(
    output_df: pd.DataFrame,
    results: BatchJudgmentResponse,
    processed_ids: set[int],
) -> None:
    """Write batch judgments into the output frame and checkpoint set."""
    id_to_index = {
        int(row_id): index for index, row_id in output_df["id"].items()
    }

    for judgment in results.results:
        row_index = id_to_index[judgment.id]
        output_df.at[row_index, "is_related"] = judgment.is_related
        processed_ids.add(judgment.id)


def main() -> None:
    load_dotenv()

    global MODEL  # noqa: PLW0603 — refresh model after .env is loaded
    MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY is not set. Add it to a .env file or environment.")
        sys.exit(1)

    if not INPUT_CSV.exists():
        print(f"Error: Input file not found: {INPUT_CSV}")
        sys.exit(1)

    max_workers = int(os.getenv("MAX_WORKERS", DEFAULT_MAX_WORKERS))

    print(f"Reading {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV)
    validate_input(df)

    processed_ids = load_checkpoint()
    output_df = load_or_init_output(df)

    total_rows = len(df)
    already_done = len(processed_ids)
    pending_batches = build_pending_batches(df, processed_ids, BATCH_SIZE)
    total_batches = len(pending_batches)

    print(f"Model: {MODEL}")
    print(f"Workers: {max_workers}")
    print(f"Total rows: {total_rows}")
    print(f"Already processed (checkpoint): {already_done}")
    print(f"Remaining batches: {total_batches} (batch size: {BATCH_SIZE})")

    if not pending_batches:
        print("Nothing to process. All rows are already in the checkpoint.")
        if USAGE_CSV.exists():
            print(f"Total spent (all runs): ${load_total_spent_usd():.6f} USD")
        return

    session_spent_usd = 0.0
    prior_spent_usd = load_total_spent_usd()
    jobs = [
        BatchJob(batch_num=index, total_batches=total_batches, rows=batch)
        for index, batch in enumerate(pending_batches, start=1)
    ]
    failures: list[tuple[BatchJob, Exception]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_batch_job, api_key, job): job for job in jobs
        }

        for future in as_completed(futures):
            job = futures[future]
            try:
                completed_job, batch_result = future.result()
                batch_cost = persist_batch_result(
                    output_df=output_df,
                    processed_ids=processed_ids,
                    batch_result=batch_result,
                    total_rows=total_rows,
                    job=completed_job,
                )
                session_spent_usd += batch_cost
            except Exception as exc:  # noqa: BLE001 — collect and continue other batches
                failures.append((job, exc))
                safe_print(
                    f"FAILED batch {job.batch_num}/{job.total_batches}: {exc}"
                )

    if failures:
        safe_print(f"\n{len(failures)} batch(es) failed:")
        for job, exc in sorted(failures, key=lambda item: item[0].batch_num):
            batch_ids = [row["id"] for row in job.rows]
            safe_print(
                f"  - batch {job.batch_num}: ids {batch_ids[0]}..{batch_ids[-1]} -> {exc}"
            )
        sys.exit(1)

    total_spent_usd = prior_spent_usd + session_spent_usd
    print(f"\nDone. Output written to {OUTPUT_CSV}")
    print(f"Usage log written to {USAGE_CSV}")
    print(f"Session cost: ${session_spent_usd:.6f} USD")
    print(f"Total cost (all runs): ${total_spent_usd:.6f} USD")


if __name__ == "__main__":
    main()
