import pandas as pd


def add_or_complete_id_column(input_csv_path, output_csv_path=None, id_column="id"):
    """
    Reads a CSV file and ensures it has a complete numeric id column.

    Rules:
    - If the id column does not exist, create it from 1 to number of rows.
    - If the id column exists, keep existing numeric ids.
    - For rows with missing/non-numeric/empty ids, assign new ids starting from max existing id + 1.
    """

    df = pd.read_csv(input_csv_path)

    if id_column not in df.columns:
        df.insert(0, id_column, range(1, len(df) + 1))
    else:
        numeric_ids = pd.to_numeric(df[id_column], errors="coerce")

        max_existing_id = numeric_ids.max()
        if pd.isna(max_existing_id):
            next_id = 1
        else:
            next_id = int(max_existing_id) + 1

        missing_mask = numeric_ids.isna()

        new_ids = range(next_id, next_id + missing_mask.sum())

        df.loc[missing_mask, id_column] = list(new_ids)

        df[id_column] = pd.to_numeric(df[id_column], errors="coerce").astype("Int64")

    if output_csv_path is None:
        output_csv_path = input_csv_path

    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

    return df


# Example usage:
add_or_complete_id_column(
    input_csv_path="job_skill.csv",
    output_csv_path="output.csv"
)