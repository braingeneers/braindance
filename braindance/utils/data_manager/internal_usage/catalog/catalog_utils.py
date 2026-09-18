#!/usr/bin/env python
"""
Catalog Utilities
==================

Utility functions for catalog processing and organoid ID assignment.
"""

import pandas as pd


def assign_organoid_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign unique organoid IDs in format: {chip_number}{letter}

    Logic:
    - Each unique chip gets a number (1, 2, 3, ...) based on alphabetical sort
    - Each unique (proj, chip) tuple gets a letter (A, B, C, ...)
    - Raises explicit error if any chip has >26 variants

    The alphabetical sorting of chips ensures stable numbering across
    catalog regenerations (as long as the set of chips doesn't change).

    Examples:
        >>> df = pd.DataFrame({
        ...     'chip': ['22097', '20247', '20247', '21985'],
        ...     'proj': ['proj1', 'proj1', 'proj2', 'proj1']
        ... })
        >>> result = assign_organoid_ids(df)
        >>> result['organoid_id'].tolist()
        ['2A', '1A', '1B', '3A']

        Sorted chips: ['20247', '21985', '22097']
        - 20247 (chip #1): proj1 → 1A, proj2 → 1B
        - 21985 (chip #2): proj1 → 2A
        - 22097 (chip #3): proj1 → 3A

    Args:
        df: DataFrame with 'chip' and 'proj' columns

    Returns:
        DataFrame with new 'organoid_id' column added

    Raises:
        KeyError: If 'chip' or 'proj' columns are missing
        ValueError: If any chip has >26 unique proj values (exceeds A-Z)
    """
    # Validate required columns
    if "chip" not in df.columns:
        raise KeyError(
            "Missing 'chip' column in DataFrame. "
            "Cannot assign organoid IDs without chip information."
        )
    if "proj" not in df.columns:
        raise KeyError(
            "Missing 'proj' column in DataFrame. "
            "Cannot assign organoid IDs without project information."
        )

    # Work on a copy to avoid modifying the original
    df = df.copy()

    # Initialize organoid_id column
    df["organoid_id"] = None

    # Sort chips alphabetically for stable numbering
    unique_chips = sorted(df["chip"].unique())

    # Track assignments for summary
    assignments = []

    for chip_idx, chip in enumerate(unique_chips, start=1):
        chip_number = chip_idx

        # Get all (proj, chip) tuples for this chip, sorted by proj
        chip_df = df[df["chip"] == chip]
        unique_projs = sorted(chip_df["proj"].dropna().unique())

        # Error if >26 variants (exceeds A-Z)
        if len(unique_projs) > 26:
            raise ValueError(
                f"Chip '{chip}' has {len(unique_projs)} unique project variants. "
                f"Maximum supported is 26 (A-Z). This likely indicates a data issue."
            )

        for letter_idx, proj in enumerate(unique_projs):
            letter = chr(ord("A") + letter_idx)  # A, B, C, ...
            organoid_id = f"{chip_number}{letter}"

            # Assign to all rows matching (proj, chip)
            mask = (df["chip"] == chip) & (df["proj"] == proj)
            df.loc[mask, "organoid_id"] = organoid_id

            # Track for summary
            n_recordings = mask.sum()
            assignments.append(
                {
                    "chip": chip,
                    "proj": proj,
                    "organoid_id": organoid_id,
                    "n_recordings": n_recordings,
                }
            )

    # Print summary
    print(f"\nAssigned organoid IDs to {len(assignments)} unique organoids:")
    print(f"  Total chips: {len(unique_chips)}")
    print(f"  Total recordings: {len(df)}")

    # Show first few assignments as examples
    print(f"\n  Examples:")
    for assignment in assignments[:5]:
        print(
            f"    {assignment['organoid_id']}: "
            f"{assignment['chip']}/{assignment['proj']} "
            f"({assignment['n_recordings']} recordings)"
        )

    if len(assignments) > 5:
        print(f"    ... and {len(assignments) - 5} more")

    return df
