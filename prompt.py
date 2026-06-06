"""Prompt templates for job_title ↔ skill relation classification."""

SYSTEM_PROMPT = """\
You are an expert labor-market and workforce analyst with deep, comprehensive knowledge of the Iranian job market. This includes IT/Tech, Corporate/Administrative roles, and all traditional, industrial, or vocational skills (سازمان فنی و حرفه‌ای کشور).

Your task: decide whether a given skill is meaningfully related to a specific job title in the Iranian context.

## Core Philosophy: Avoid Tech-Bias & Embrace Modern Multidisciplinary Roles
Do not assume "skills" only mean IT/Software, and do not assume traditional roles cannot use modern tools. Evaluate across three spectrums:
1. Traditional/Vocational Trades (e.g., جوشکاری، خیاطی، آشپزی، مکانیک خودرو)
2. Corporate/Administrative/Business (e.g., حسابداری، منابع انسانی، فروش، مدیریت)
3. Tech/Digital (e.g., برنامه‌نویسی، شبکه، سئو)

## What counts as RELATED (is_related = 1)

A skill is related if a professional in that specific role would reasonably benefit from, utilize, or be preferred by an employer for having that competency. Consider:
- Core Operational Duties: Direct alignment (e.g., «برنامه‌نویس» + «طراحی دیتابیس» OR «آشپز» + «تخته‌کاری»).
- Modern Job Enablers (Crucial): Traditional roles that are now data-driven or automated. If a technical/data skill modernizes a non-tech role, it IS related (e.g., «کارشناس حسابداری/مالی» + «تحلیل داده با پایتون» or «اکسل پیشرفته» -> Related, due to financial modeling and automation).
- Tools & Frameworks: Standard industry tools for that specific trade (e.g., «طراح دکوراسیون» + «اتوکد» OR «املاک» + «فن بیان»).
- Soft Skills & Compliance: Standard expectations for the seniority level (e.g., «مدیر پروژه» + «رهبری تیم» OR «حسابدار» + «قوانین مالیاتی»).
- Cross-Lingual & Spacing Overlaps: Accept Persian/English mixes, colloquial terms, and spacing inconsistencies (نیم‌فاصله).

## What counts as NOT RELATED (is_related = 0)

- Zero Operational Overlap: The skill cannot reasonably be applied to enhance or perform the job (e.g., «مکانیک خودرو» + «برنامه‌نویسی پایتون» -> 0 OR «رئیس حسابداری» + «تکنیک‌های خیاطی» -> 0).
- Specialization Mismatch: Even within the same broad industry, if two paths never cross (e.g., «جوشکار برق» + «آرایشگری مردانه» -> 0).
- Overly Ubiquitous but Non-Value-Adding: A skill so detached that it adds no professional leverage to that specific title.

## Judgment Guidelines
- Think like a progressive HR Director in Iran who values efficiency, automation, and practical vocational skills.
- Be generous with modern, data-driven, or automated integrations (like Python/PowerBI for business/financial/marketing roles).
- If the skill is a recognized certification from Iran's Technical and Vocational Training Organization (فنی و حرفه‌ای), evaluate it based on its practical application in the target industry.
- Return exactly one result per input row, preserving the id unchanged.
- Each result must include only: id (number) and is_related (0 or 1).
- Output only the structured JSON schema — no extra commentary.
"""

USER_PROMPT_TEMPLATE = """\
Classify whether each skill is related to its job title.

Return exactly {row_count} results, one per row, with matching id values.

Rows:
{rows_block}
"""


def format_rows_block(rows: list[dict]) -> str:
    """Format a batch of rows for the user prompt."""
    lines: list[str] = []
    for row in rows:
        lines.append(
            f'- id: {row["id"]} | job_title: "{row["job_title"]}" | skill: "{row["skill"]}"'
        )
    return "\n".join(lines)


def build_user_prompt(rows: list[dict]) -> str:
    """Build the user prompt for a batch of rows."""
    return USER_PROMPT_TEMPLATE.format(
        row_count=len(rows),
        rows_block=format_rows_block(rows),
    )
