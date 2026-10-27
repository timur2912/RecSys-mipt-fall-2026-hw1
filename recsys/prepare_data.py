"""Create a reproducible Avito ML Cup subset for recommender experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/avito"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/avito"),
    )
    parser.add_argument("--user-modulus", type=int, default=32)
    parser.add_argument("--core-size", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    """Build and save the deterministic user-level subset."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clickstream_path = args.raw_dir / "clickstream.pq"
    events_path = args.raw_dir / "events.pq"
    features_path = args.raw_dir / "cat_features.pq"
    sample_path = args.output_dir / "interactions_sample.parquet"
    interactions_path = args.output_dir / "interactions.parquet"
    metadata_path = args.output_dir / "item_metadata.parquet"

    for path in (clickstream_path, events_path, features_path):
        if not path.exists():
            raise FileNotFoundError(f"Required input file is missing: {path}")

    connection = duckdb.connect()
    connection.execute("SET threads TO 4")
    connection.execute("SET memory_limit = '8GB'")
    connection.execute(
        f"""
        COPY (
            SELECT
                c.cookie AS user_id,
                c.node::BIGINT AS item_id,
                MAX(c.event_date) AS timestamp,
                MAX(COALESCE(e.is_contact, 0))::TINYINT AS contact
            FROM read_parquet('{clickstream_path.as_posix()}') AS c
            LEFT JOIN read_parquet('{events_path.as_posix()}') AS e USING (event)
            WHERE c.cookie % {args.user_modulus} = 0
            GROUP BY c.cookie, c.node
        ) TO '{sample_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    interactions = pd.read_parquet(sample_path)
    while True:
        previous_size = len(interactions)
        user_counts = interactions["user_id"].value_counts()
        item_counts = interactions["item_id"].value_counts()
        interactions = interactions[
            interactions["user_id"].isin(
                user_counts[user_counts >= args.core_size].index
            )
            & interactions["item_id"].isin(
                item_counts[item_counts >= args.core_size].index
            )
        ].copy()
        if len(interactions) == previous_size:
            break

    interactions.to_parquet(interactions_path, index=False)
    connection.execute(
        f"""
        COPY (
            WITH selected_nodes AS (
                SELECT DISTINCT item_id
                FROM read_parquet('{interactions_path.as_posix()}')
            )
            SELECT
                c.node::BIGINT AS item_id,
                MODE(c.category)::BIGINT AS category,
                MODE(c.location)::BIGINT AS location,
                COUNT(*)::BIGINT AS listing_count
            FROM read_parquet('{features_path.as_posix()}') AS c
            INNER JOIN selected_nodes AS s ON c.node = s.item_id
            GROUP BY c.node
        ) TO '{metadata_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    print(f"Interactions: {len(interactions):,}")
    print(f"Users: {interactions['user_id'].nunique():,}")
    print(f"Items: {interactions['item_id'].nunique():,}")
    print(f"Contacts: {interactions['contact'].sum():,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
