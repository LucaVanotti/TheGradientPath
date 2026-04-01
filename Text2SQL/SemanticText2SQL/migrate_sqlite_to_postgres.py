#!/usr/bin/env python3
"""
Migrate `offerte_ristorazione.db` (SQLite) to PostgreSQL.

Default source:
  ../offerte_ristorazione.db

Target connection can be provided with either:
- --pg-dsn "postgresql://user:pass@host:5432/dbname"
- PG_DSN env var
- individual env vars: PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD

This script is idempotent for table `prodotti` by using upsert on primary key `id`.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path
from typing import Iterable

import psycopg2
from psycopg2.extras import execute_values


SQL_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS prodotti (
    id TEXT PRIMARY KEY,
    nome TEXT,
    fornitore TEXT,
    codice TEXT,
    prezzo DOUBLE PRECISION,
    cliente TEXT,
    offerta_num TEXT,
    specifiche TEXT,
    descrizione TEXT,
    data_offerta TEXT
);
"""

SQL_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_prodotti_codice ON prodotti(codice);",
    "CREATE INDEX IF NOT EXISTS idx_prodotti_cliente ON prodotti(cliente);",
    "CREATE INDEX IF NOT EXISTS idx_prodotti_offerta_num ON prodotti(offerta_num);",
    "CREATE INDEX IF NOT EXISTS idx_prodotti_data_offerta ON prodotti(data_offerta);",
]

SQL_SELECT_SQLITE = """
SELECT
    id,
    nome,
    fornitore,
    codice,
    prezzo,
    cliente,
    offerta_num,
    specifiche,
    descrizione,
    data_offerta
FROM prodotti
"""

SQL_INSERT_POSTGRES = """
INSERT INTO prodotti (
    id,
    nome,
    fornitore,
    codice,
    prezzo,
    cliente,
    offerta_num,
    specifiche,
    descrizione,
    data_offerta
)
VALUES %s
ON CONFLICT (id) DO UPDATE SET
    nome = EXCLUDED.nome,
    fornitore = EXCLUDED.fornitore,
    codice = EXCLUDED.codice,
    prezzo = EXCLUDED.prezzo,
    cliente = EXCLUDED.cliente,
    offerta_num = EXCLUDED.offerta_num,
    specifiche = EXCLUDED.specifiche,
    descrizione = EXCLUDED.descrizione,
    data_offerta = EXCLUDED.data_offerta;
"""


def build_pg_conn_kwargs(args: argparse.Namespace) -> dict:
    if args.pg_dsn:
        return {"dsn": args.pg_dsn}

    pg_dsn_env = os.getenv("PG_DSN")
    if pg_dsn_env:
        return {"dsn": pg_dsn_env}

    host = os.getenv("PGHOST", args.pg_host)
    port = int(os.getenv("PGPORT", str(args.pg_port)))
    database = os.getenv("PGDATABASE", args.pg_database)
    user = os.getenv("PGUSER", args.pg_user)
    password = os.getenv("PGPASSWORD", args.pg_password)

    if not all([host, port, database, user, password]):
        raise ValueError(
            "Missing PostgreSQL connection values. Use --pg-dsn or set PG_DSN, "
            "or provide PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD."
        )

    return {
        "host": host,
        "port": port,
        "database": database,
        "user": user,
        "password": password,
    }


def sqlite_rows(sqlite_path: Path) -> Iterable[tuple]:
    conn = sqlite3.connect(str(sqlite_path))
    try:
        cursor = conn.cursor()
        cursor.execute(SQL_SELECT_SQLITE)
        for row in cursor.fetchall():
            yield row
    finally:
        conn.close()


def sqlite_count(sqlite_path: Path) -> int:
    conn = sqlite3.connect(str(sqlite_path))
    try:
        cursor = conn.cursor()
        return int(cursor.execute("SELECT COUNT(*) FROM prodotti").fetchone()[0])
    finally:
        conn.close()


def pg_count(pg_conn) -> int:
    with pg_conn.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM prodotti")
        return int(cursor.fetchone()[0])


def migrate(args: argparse.Namespace) -> None:
    sqlite_path = Path(args.sqlite_path).resolve()
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite file not found: {sqlite_path}")

    source_count = sqlite_count(sqlite_path)
    print(f"[1/4] Source SQLite rows in prodotti: {source_count}")

    pg_kwargs = build_pg_conn_kwargs(args)

    if args.dry_run:
        print("[2/4] Dry run mode enabled: no PostgreSQL changes will be executed.")
        print(f"SQLite path: {sqlite_path}")
        print(f"PostgreSQL connection args (sanitized): { {k: ('***' if k == 'password' else v) for k, v in pg_kwargs.items()} }")
        return

    pg_conn = psycopg2.connect(**pg_kwargs)
    try:
        pg_conn.autocommit = False

        with pg_conn.cursor() as cursor:
            print("[2/4] Creating target table and indexes if not present...")
            cursor.execute(SQL_CREATE_TABLE)
            for sql in SQL_CREATE_INDEXES:
                cursor.execute(sql)

            if args.truncate_target:
                print("[2/4] Truncating target table prodotti...")
                cursor.execute("TRUNCATE TABLE prodotti")

        rows = list(sqlite_rows(sqlite_path))
        print(f"[3/4] Upserting {len(rows)} rows into PostgreSQL...")

        with pg_conn.cursor() as cursor:
            if rows:
                execute_values(cursor, SQL_INSERT_POSTGRES, rows, page_size=500)

        pg_conn.commit()

        target_count = pg_count(pg_conn)
        print(f"[4/4] PostgreSQL rows in prodotti after migration: {target_count}")

        if target_count < source_count:
            raise RuntimeError(
                "Post-migration validation failed: target row count is lower than source."
            )

        print("Migration completed successfully.")
    except Exception:
        pg_conn.rollback()
        raise
    finally:
        pg_conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate SQLite offerte_ristorazione to PostgreSQL")
    parser.add_argument(
        "--sqlite-path",
        default=str((Path(__file__).resolve().parent.parent / "offerte_ristorazione.db")),
        help="Path to SQLite file (default: ../offerte_ristorazione.db)",
    )
    parser.add_argument("--pg-dsn", default=None, help="PostgreSQL DSN, e.g. postgresql://user:pass@host:5432/db")
    parser.add_argument("--pg-host", default="localhost", help="PostgreSQL host if DSN is not used")
    parser.add_argument("--pg-port", default=5432, type=int, help="PostgreSQL port if DSN is not used")
    parser.add_argument("--pg-database", default="offerte_ristorazione", help="PostgreSQL database name")
    parser.add_argument("--pg-user", default="bookadmin", help="PostgreSQL user")
    parser.add_argument("--pg-password", default="bookpass123", help="PostgreSQL password")
    parser.add_argument("--truncate-target", action="store_true", help="Truncate target table before loading")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print plan only")
    return parser.parse_args()


if __name__ == "__main__":
    migrate(parse_args())
