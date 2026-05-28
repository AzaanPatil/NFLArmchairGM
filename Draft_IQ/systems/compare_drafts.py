def compare_drafts(mock_df, real_df):
    merged = mock_df.merge(
        real_df,
        on=["Year", "Player"],
        suffixes=("_mock", "_actual")
    )

    merged["pick_diff"] = abs(
        merged["Pick_mock"].astype(int) - merged["Pick_actual"].astype(int)
    )

    return merged