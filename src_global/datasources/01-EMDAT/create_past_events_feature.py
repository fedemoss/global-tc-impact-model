#!/usr/bin/env python3
import os

import numpy as np
import pandas as pd


def count_events_in_last_5_years(
    df,
    year_col="Start Year",
    month_col="Start Month",
    country_col="GID_0",
    event_col="DisNo.",
):
    """
    For each event, counts the number of events in the same country that happened in the last 5 years,
    considering both the year and month. If there are fewer than 5 years of data, computes the cumulative sum
    until we can compute the 5 years gap.

    Parameters:
    df (pd.DataFrame): DataFrame containing event data with country, year, and month columns.
    year_col (str): Column name for the event year.
    month_col (str): Column name for the event month.
    country_col (str): Column name for the country.
    event_col (str): Column name for event identification.

    Returns:
    pd.DataFrame: A DataFrame with country-wise event counts for each event.
    """

    # Function to get the year and month as a datetime object for easier comparison
    def to_datetime(year, month):
        return pd.to_datetime(f"{int(year)}-{int(month)}", format="%Y-%m")

    # Get the latest date in the dataset considering year and month
    df["Event Date"] = df.apply(
        lambda row: to_datetime(row[year_col], row[month_col]), axis=1
    )
    latest_date = df["Event Date"].max()

    # Function to count events for each row
    def count_events_for_row(row):
        # Get the subset of events for the same country
        country_data = df[df[country_col] == row[country_col]]

        # Get the event's date
        event_date = row["Event Date"]

        # Define the 5-year window by subtracting 5 years from the event date
        five_years_ago = event_date - pd.DateOffset(years=5)

        # Filter the events that occurred within the last 5 years considering both year and month
        last_5_years_data = country_data[
            (country_data["Event Date"] >= five_years_ago)
            & (country_data["Event Date"] <= event_date)
        ]

        # Return the count of events in the last 5 years (including the event itself)
        return last_5_years_data.shape[0]

    # Apply the function to each row in the dataframe
    df["N_events_5_years"] = df.apply(count_events_for_row, axis=1)

    return df[
        [event_col, country_col, year_col, month_col, "N_events_5_years"]
    ]


if __main__ == "__main__":
    # Load EMDAT data
    emdat_data = pd.read_csv("/data/big/fmoss/data/EMDAT/impact_data.csv")
    temporal_data = emdat_data[
        ["DisNo.", "GID_0", "Start Year", "Start Month"]
    ].drop_duplicates(ignore_index=True)
    # Count events
    last_5_years = count_events_in_last_5_years(temporal_data)
    # Save the output
    last_5_years.to_csv(
        "/data/big/fmoss/data/EMDAT/impact_in_last_5_years_data.csv",
        index=False,
    )
