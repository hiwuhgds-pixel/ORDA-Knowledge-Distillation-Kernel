from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

Record = Mapping[str, object]


# ── Column definition ────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Column:
    key: str
    header: str
    align_right: bool = False
    # Console rendering; defaults to str(). The CSV always writes the raw
    # value, so it keeps full precision while the console rounds.
    fmt: Callable[[object], str] | None = None
    csv_header: str | None = None

    def console_cell(self, record: Record) -> str:
        value = record[self.key]
        return self.fmt(value) if self.fmt is not None else str(value)

    def csv_cell(self, record: Record):
        return record[self.key]


# ── Table schema ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TableSchema:
    columns: tuple[Column, ...]
    title: str | None = None
    separate_on: str | None = None

    def select(self, *keys: str, title: str | None = None) -> TableSchema:
        by_key = {column.key: column for column in self.columns}
        return TableSchema(tuple(by_key[key] for key in keys), title=title)

    @property
    def headers(self) -> list[str]:
        return [column.header for column in self.columns]

    @property
    def csv_headers(self) -> list[str]:
        return [column.csv_header or column.header for column in self.columns]

    @property
    def align_right(self) -> set[int]:
        return {
            index for index, column in enumerate(self.columns) if column.align_right
        }

    @property
    def separate_index(self) -> int | None:
        if self.separate_on is None:
            return None
        return next(
            index
            for index, column in enumerate(self.columns)
            if column.key == self.separate_on
        )

    def console_rows(self, records: Sequence[Record]) -> list[list[str]]:
        return [
            [column.console_cell(record) for column in self.columns]
            for record in records
        ]

    def csv_rows(self, records: Sequence[Record]) -> list[list[object]]:
        return [
            [column.csv_cell(record) for column in self.columns] for record in records
        ]


# ── Record storage ───────────────────────────────────────────────────────────
@dataclass
class RecordTable:
    schema: TableSchema
    records: list[dict[str, object]] = field(default_factory=list)

    def add(self, **values: object) -> dict[str, object]:
        expected = {column.key for column in self.schema.columns}
        missing = expected - values.keys()
        if missing:
            raise KeyError(f"record is missing columns: {sorted(missing)}")
        unexpected = values.keys() - expected
        if unexpected:
            raise KeyError(f"record has unknown columns: {sorted(unexpected)}")
        self.records.append(dict(values))
        return self.records[-1]

    def extend(self, records: Sequence[Record]) -> None:
        for record in records:
            self.add(**record)

    def filter(self, **equals: object) -> list[dict[str, object]]:
        return [
            record
            for record in self.records
            if all(record[key] == value for key, value in equals.items())
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)
