import pandas as pd
import numpy as np

def split_csv(
    input_csv: str,
    output_specs: list[tuple[str, float]],
    shuffle_seed: int = 42
) -> None:
    """
    Randomly divide a CSV into multiple files by percentage.

    Parameters
    ----------
    input_csv : str
        Path to the source CSV file.
    output_specs : list of (filename, percentage)
        Each tuple contains:
          - filename (str): e.g. "part1.csv"
          - percentage (float): between 0 and 1 (e.g. 0.3 for 30%).
        Percentages must sum to <= 1.0
    shuffle_seed : int, optional
        Random seed for reproducibility.
    """

    df = pd.read_csv(input_csv)
    df_shuffled = df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)

    # ensure percentages add up to at most 1.0
    assert sum(p for _, p in output_specs) <= 1.0, "Percentages exceed 100%"

    start_idx = 0
    n_rows = len(df_shuffled)

    for filename, pct in output_specs:
        end_idx = start_idx + int(round(pct * n_rows))
        df_part = df_shuffled.iloc[start_idx:end_idx]
        df_part.to_csv(filename, index=False)
        print(f"Saved {len(df_part)} rows to {filename}")
        start_idx = end_idx

    # if any remainder rows exist (because of rounding), save them
    if start_idx < n_rows:
        remainder = df_shuffled.iloc[start_idx:]
        remainder_name = "remainder.csv"
        remainder.to_csv(remainder_name, index=False)
        print(f"Saved remaining {len(remainder)} rows to {remainder_name}")

def combine_csv(csv1: str, csv2: str, output_csv: str) -> None:
    """
    Combine two CSV files vertically into one.

    Parameters
    ----------
    csv1 : str
        Path to the first CSV file.
    csv2 : str
        Path to the second CSV file.
    output_csv : str
        Path to save the combined CSV.
    """

    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    # Combine vertically, ignoring index to reset it
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # Optional: Shuffle combined data to mix rows from both
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    combined_df.to_csv(output_csv, index=False)
    print(f"Global Dataset CSV saved with {len(combined_df)} rows to {output_csv}")

##########################################################################################

# split_csv(
#     input_csv="selected_features_dataset.csv",
#     output_specs = [
#         ("global_train.csv", 0.6),
#         ("clients.csv", 0.3),
#         ("global_test.csv", 0.1),
#     ],
#     shuffle_seed=123
# )

# split_csv(
#     input_csv="global_train.csv",
#     output_specs = [
#         ("train.csv", 0.8),
#         ("val.csv", 0.2),
#     ],
#     shuffle_seed=123
# )

# split_csv(
#     input_csv="clients.csv",
#     output_specs = [
#     ("client1.csv", 0.116),
#     ("client2.csv", 0.1),
#     ("client3.csv", 0.088),
#     ("client4.csv", 0.156),
#     ("client5.csv", 0.113),
#     ("client6.csv", 0.096),
#     ("client7.csv", 0.065),
#     ("client8.csv", 0.042),
#     ("client9.csv", 0.132),
#     ("client10.csv", 0.092),
#     ],
#     shuffle_seed=123
# )

