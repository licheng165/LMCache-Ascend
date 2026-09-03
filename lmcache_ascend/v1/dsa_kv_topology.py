# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Any


DSAKVRowKey = tuple[str, int, int, int, int]
DSAExecutionKey = tuple[int, DSAKVRowKey, DSAKVRowKey | None]


@dataclass(frozen=True)
class DSAKVTopologyView:
    """Validated, immutable projection of vLLM's DSA KV topology."""

    signature: str
    rows_by_group: tuple[tuple[DSAKVRowKey, ...], tuple[DSAKVRowKey, ...]]
    executions: tuple[DSAExecutionKey, ...]

    @property
    def layer_counts(self) -> tuple[int, int]:
        return len(self.rows_by_group[0]), len(self.rows_by_group[1])

    def layer_indices(self, kv_group: int) -> tuple[int, ...]:
        if kv_group not in (0, 1):
            raise ValueError(
                f"DSA KV topology only supports kv_group 0 or 1, got {kv_group}."
            )
        return tuple(row[3] for row in self.rows_by_group[kv_group])


def _field(value: Any, name: str, context: str) -> Any:
    try:
        return getattr(value, name)
    except AttributeError as exc:
        raise ValueError(f"DSA KV topology {context} is missing {name!r}.") from exc


def _integer(value: Any, name: str, context: str) -> int:
    result = _field(value, name, context)
    if not isinstance(result, int) or isinstance(result, bool):
        raise ValueError(
            f"DSA KV topology {context} has invalid {name}={result!r}; "
            "expected an integer."
        )
    return result


def _row_key(row: Any, expected_group: int, position: int) -> DSAKVRowKey:
    context = f"group {expected_group} row {position}"
    layer_name = _field(row, "layer_name", context)
    if not isinstance(layer_name, str) or not layer_name:
        raise ValueError(
            f"DSA KV topology {context} has invalid layer_name={layer_name!r}."
        )
    execution_ordinal = _integer(row, "execution_ordinal", context)
    kv_group = _integer(row, "kv_group", context)
    row_ordinal = _integer(row, "row_ordinal", context)
    bank = _integer(row, "bank", context)
    if execution_ordinal < 0:
        raise ValueError(
            f"DSA KV topology {context} has negative execution_ordinal="
            f"{execution_ordinal}."
        )
    if kv_group != expected_group:
        raise ValueError(f"DSA KV topology {context} declares kv_group={kv_group}.")
    if row_ordinal != position:
        raise ValueError(
            f"DSA KV topology group {expected_group} row ordinals must be dense; "
            f"position={position} row_ordinal={row_ordinal}."
        )
    if bank != row_ordinal % 2:
        raise ValueError(
            f"DSA KV topology {context} has bank={bank}; expected "
            f"row_ordinal % 2 = {row_ordinal % 2}."
        )
    return layer_name, execution_ordinal, kv_group, row_ordinal, bank


def validate_dsa_kv_topology(topology: Any) -> DSAKVTopologyView:
    """Validate and project the immutable vLLM DSA topology descriptor."""
    if topology is None:
        raise ValueError("DSA KV topology descriptor is missing.")

    signature = _field(topology, "signature", "descriptor")
    if not isinstance(signature, str) or not signature.strip():
        raise ValueError(
            f"DSA KV topology has invalid signature={signature!r}; "
            "expected a non-empty string."
        )

    raw_groups = _field(topology, "rows_by_group", "descriptor")
    if not isinstance(raw_groups, tuple) or len(raw_groups) != 2:
        raise ValueError(
            "DSA KV topology rows_by_group must be an immutable two-group tuple."
        )

    groups: list[tuple[DSAKVRowKey, ...]] = []
    layer_names: set[str] = set()
    for kv_group, raw_rows in enumerate(raw_groups):
        if not isinstance(raw_rows, tuple) or not raw_rows:
            raise ValueError(
                "DSA KV topology requires a non-empty immutable row tuple for "
                f"kv_group={kv_group}."
            )
        rows = tuple(
            _row_key(row, kv_group, position) for position, row in enumerate(raw_rows)
        )
        row_names = tuple(row[0] for row in rows)
        duplicate_names = layer_names.intersection(row_names)
        duplicate_names.update(name for name in row_names if row_names.count(name) > 1)
        if duplicate_names:
            raise ValueError(
                "DSA KV topology layer names must be globally unique; duplicates="
                f"{sorted(duplicate_names)}."
            )
        layer_names.update(row[0] for row in rows)
        groups.append(rows)

    latent_by_execution = {row[1]: row for row in groups[0]}
    if len(latent_by_execution) != len(groups[0]):
        raise ValueError("DSA KV topology has duplicate LATENT execution ordinals.")
    expected_executions = tuple(range(len(groups[0])))
    if tuple(sorted(latent_by_execution)) != expected_executions:
        raise ValueError(
            "DSA KV topology LATENT execution ordinals must be dense; "
            f"expected={expected_executions} "
            f"actual={tuple(sorted(latent_by_execution))}."
        )

    indexer_by_execution = {row[1]: row for row in groups[1]}
    if len(indexer_by_execution) != len(groups[1]):
        raise ValueError("DSA KV topology has duplicate INDEXER execution ordinals.")
    orphan_indexers = sorted(set(indexer_by_execution) - set(latent_by_execution))
    if orphan_indexers:
        raise ValueError(
            "DSA KV topology has INDEXER rows without exact LATENT siblings: "
            f"executions={orphan_indexers}."
        )

    raw_executions = _field(topology, "executions", "descriptor")
    if not isinstance(raw_executions, tuple):
        raise ValueError("DSA KV topology executions must be an immutable tuple.")
    if len(raw_executions) != len(groups[0]):
        raise ValueError(
            "DSA KV topology execution/layout mismatch: "
            f"executions={len(raw_executions)} latent_rows={len(groups[0])}."
        )

    executions: list[DSAExecutionKey] = []
    for position, execution in enumerate(raw_executions):
        context = f"execution {position}"
        execution_ordinal = _integer(execution, "execution_ordinal", context)
        if execution_ordinal != position:
            raise ValueError(
                "DSA KV topology execution ordinals must be dense; "
                f"position={position} execution_ordinal={execution_ordinal}."
            )

        latent = _row_key(_field(execution, "latent", context), 0, position)
        expected_latent = latent_by_execution[position]
        if latent != expected_latent:
            raise ValueError(
                "DSA KV topology execution/layout mismatch for LATENT row: "
                f"execution={position} execution_row={latent} "
                f"group_row={expected_latent}."
            )

        raw_indexer = _field(execution, "indexer", context)
        expected_indexer = indexer_by_execution.get(position)
        if raw_indexer is None:
            indexer = None
        else:
            indexer_position = _integer(raw_indexer, "row_ordinal", context)
            indexer = _row_key(raw_indexer, 1, indexer_position)
        if indexer != expected_indexer:
            raise ValueError(
                "DSA KV topology execution/layout mismatch for INDEXER row: "
                f"execution={position} execution_row={indexer} "
                f"group_row={expected_indexer}."
            )
        executions.append((execution_ordinal, latent, indexer))

    return DSAKVTopologyView(
        signature=signature,
        rows_by_group=(groups[0], groups[1]),
        executions=tuple(executions),
    )


def validate_matching_dsa_kv_topologies(
    expected: DSAKVTopologyView,
    actual: DSAKVTopologyView,
    *,
    expected_owner: str,
    actual_owner: str,
) -> None:
    """Fail closed when two construction-path topology caches disagree."""
    if expected.signature != actual.signature:
        raise ValueError(
            "DSA KV topology signature mismatch: "
            f"{expected_owner}={expected.signature} "
            f"{actual_owner}={actual.signature}."
        )
    if (
        expected.rows_by_group != actual.rows_by_group
        or expected.executions != actual.executions
    ):
        raise ValueError(
            "DSA KV topology layout mismatch despite matching signature: "
            f"{expected_owner} and {actual_owner} describe different rows."
        )
