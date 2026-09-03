# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache_ascend.integration.vllm.vllm_v1_adapter import (
    LMCacheAscendConnectorV1Impl,
)
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine
from lmcache_ascend.v1.dsa_kv_topology import validate_dsa_kv_topology
from lmcache_ascend.v1.npu_connector.npu_connectors import (
    _GroupLayout,
    VLLMPagedMemLayerwiseNPUConnector,
)


LATENT_LAYERS = 79
INDEXER_EXECUTIONS = (
    0,
    1,
    2,
    6,
    10,
    14,
    18,
    22,
    26,
    30,
    34,
    38,
    42,
    46,
    50,
    54,
    58,
    62,
    66,
    70,
    74,
    78,
)


def _row(
    layer_name: str,
    execution_ordinal: int,
    kv_group: int,
    row_ordinal: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        layer_name=layer_name,
        execution_ordinal=execution_ordinal,
        kv_group=kv_group,
        row_ordinal=row_ordinal,
        bank=row_ordinal % 2,
    )


def _topology(
    signature: str = "glm52-topology-signature",
    name_prefix: str = "physical",
) -> SimpleNamespace:
    latent = tuple(
        _row(f"{name_prefix}.latent.{execution}", execution, 0, execution)
        for execution in range(LATENT_LAYERS)
    )
    indexer = tuple(
        _row(f"{name_prefix}.sparse.{execution}", execution, 1, row)
        for row, execution in enumerate(INDEXER_EXECUTIONS)
    )
    indexer_by_execution = {row.execution_ordinal: row for row in indexer}
    executions = tuple(
        SimpleNamespace(
            execution_ordinal=execution,
            latent=latent[execution],
            indexer=indexer_by_execution.get(execution),
        )
        for execution in range(LATENT_LAYERS)
    )
    return SimpleNamespace(
        executions=executions,
        rows_by_group=(latent, indexer),
        signature=signature,
    )


def test_glm52_topology_79_22_execution_goldens() -> None:
    view = validate_dsa_kv_topology(_topology())

    assert view.layer_counts == (79, 22)
    assert view.layer_indices(0) == tuple(range(79))
    assert view.layer_indices(1) == tuple(range(22))
    assert view.executions[6][1][3:] == (6, 0)
    assert view.executions[6][2] is not None
    assert view.executions[6][2][3:] == (3, 1)
    assert view.executions[78][1][3:] == (78, 0)
    assert view.executions[78][2] is not None
    assert view.executions[78][2][3:] == (21, 1)
    assert view.executions[3][2] is None


def test_adapter_uses_descriptor_rows_without_name_role_inference() -> None:
    topology = _topology()
    groups = tuple(
        SimpleNamespace(layer_names=[row.layer_name for row in rows])
        for rows in topology.rows_by_group
    )
    kv_cache_config = SimpleNamespace(
        dsa_kv_topology=topology,
        kv_cache_groups=groups,
    )
    adapter = object.__new__(LMCacheAscendConnectorV1Impl)
    adapter.dsa_kv_topology = topology
    adapter._dsa_kv_topology_view = None

    counts = adapter._derive_runtime_kv_group_layer_counts(True, kv_cache_config)
    assert counts == (79, 22)

    all_names = [row.layer_name for rows in topology.rows_by_group for row in rows]
    adapter.config = SimpleNamespace(dsa_two_groups=True)
    adapter.kv_caches = {name: object() for name in reversed(all_names)}
    adapter._refresh_kvcaches_list()

    assert adapter._latent_layer_names == [
        row.layer_name for row in topology.rows_by_group[0]
    ]
    assert adapter._indexer_layer_names == [
        row.layer_name for row in topology.rows_by_group[1]
    ]
    assert len(adapter._latent_kvcaches) == 79
    assert len(adapter._indexer_kvcaches) == 22


@pytest.mark.parametrize(
    "mutate",
    [
        lambda topology: setattr(topology.rows_by_group[1][3], "bank", 0),
        lambda topology: setattr(topology, "signature", ""),
        lambda topology: setattr(
            topology.executions[6], "indexer", topology.rows_by_group[1][2]
        ),
    ],
    ids=("bad-bank", "empty-signature", "execution-layout"),
)
def test_provided_malformed_topology_fails_closed(mutate) -> None:
    topology = _topology()
    mutate(topology)
    adapter = object.__new__(LMCacheAscendConnectorV1Impl)
    adapter.dsa_kv_topology = topology
    adapter._dsa_kv_topology_view = None

    with pytest.raises(ValueError, match="DSA KV topology"):
        adapter._derive_runtime_kv_group_layer_counts(
            True,
            SimpleNamespace(dsa_kv_topology=topology, kv_cache_groups=None),
        )


def _d_node_adapter(
    monkeypatch,
    *,
    use_layerwise: bool = True,
    enable_sparse_attention: bool = True,
    supports_cold_compact: bool = True,
    shrink: str = "2",
    unbundle: str = "1",
    two_groups: str = "1",
    shared_pool: str = "1",
) -> LMCacheAscendConnectorV1Impl:
    adapter = object.__new__(LMCacheAscendConnectorV1Impl)
    adapter.config = SimpleNamespace(
        use_layerwise=use_layerwise,
        enable_sparse_attention=enable_sparse_attention,
    )
    adapter.supports_dsa_cold_compact_load = lambda: supports_cold_compact
    monkeypatch.setenv("VLLM_ASCEND_DSA_UNBUNDLE", unbundle)
    monkeypatch.setenv("VLLM_ASCEND_DSA_TWO_GROUPS", two_groups)
    monkeypatch.setenv("VLLM_ASCEND_DSA_SHARED_POOL", shared_pool)
    monkeypatch.setenv("VLLM_ASCEND_DSA_SHRINK_LATENT", shrink)
    return adapter


def _d_node_vllm_config() -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
    )


def test_d_node_prerequisites_pass_when_closed_set_is_satisfied(monkeypatch) -> None:
    adapter = _d_node_adapter(monkeypatch)

    adapter._validate_sparse_decode_d_node_prerequisites(_d_node_vllm_config())


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"shrink": "0"}, "DSA_SHRINK_LATENT=2"),
        ({"use_layerwise": False}, "use_layerwise=true"),
        ({"enable_sparse_attention": False}, "enable_sparse_attention=true"),
        ({"supports_cold_compact": False}, "supports_dsa_compact_external_load"),
        ({"unbundle": "0"}, "DSA_UNBUNDLE=1"),
    ],
)
def test_d_node_prerequisites_fail_closed_per_condition(
    monkeypatch,
    kwargs,
    expected,
) -> None:
    adapter = _d_node_adapter(monkeypatch, **kwargs)

    with pytest.raises(ValueError, match=expected):
        adapter._validate_sparse_decode_d_node_prerequisites(
            _d_node_vllm_config()
        )


def test_d_node_env_uses_strict_boolean_spelling(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_ASCEND_DSA_SPARSE_DECODE_D_NODE", raising=False)
    assert LMCacheAscendConnectorV1Impl._sparse_decode_d_node_requested() is False
    monkeypatch.setenv("VLLM_ASCEND_DSA_SPARSE_DECODE_D_NODE", " TrUe ")
    assert LMCacheAscendConnectorV1Impl._sparse_decode_d_node_requested() is True
    monkeypatch.setenv("VLLM_ASCEND_DSA_SPARSE_DECODE_D_NODE", "1")
    with pytest.raises(ValueError, match="must be 'true' or 'false'"):
        LMCacheAscendConnectorV1Impl._sparse_decode_d_node_requested()


def test_group_layout_is_constructed_from_descriptor_cardinality() -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.dsa_two_groups = True
    connector._group_layouts = {}
    connector.dsa_kv_topology = None
    connector._dsa_kv_topology_view = None
    connector.cache_dsa_kv_topology(_topology())

    latent_layout = _GroupLayout()
    indexer_layout = _GroupLayout()
    connector._apply_topology_group_layout(0, latent_layout, actual_layers=79)
    connector._apply_topology_group_layout(1, indexer_layout, actual_layers=22)

    assert latent_layout.num_layers == 79
    assert latent_layout.layer_indices == tuple(range(79))
    assert indexer_layout.num_layers == 22
    assert indexer_layout.layer_indices == tuple(range(22))
    assert latent_layout.topology_signature == _topology().signature
    assert indexer_layout.topology_signature == _topology().signature


def test_connector_rejects_signature_and_layout_mismatch() -> None:
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.dsa_two_groups = True
    connector._group_layouts = {}
    connector.dsa_kv_topology = None
    connector._dsa_kv_topology_view = None
    connector.cache_dsa_kv_topology(_topology(signature="signature-a"))

    with pytest.raises(ValueError, match="signature mismatch"):
        connector.cache_dsa_kv_topology(_topology(signature="signature-b"))
    with pytest.raises(ValueError, match="layout mismatch"):
        connector.cache_dsa_kv_topology(
            _topology(signature="signature-a", name_prefix="different")
        )

    wrong_layout = _GroupLayout()
    wrong_layout.num_layers = 79
    wrong_layout.layer_indices = tuple(range(79))
    connector._group_layouts = {1: wrong_layout}
    with pytest.raises(ValueError, match="layout mismatch"):
        connector.cache_dsa_kv_topology(_topology(signature="signature-a"))


def test_engine_caches_same_descriptor_on_metadata_and_connector() -> None:
    topology = _topology()
    received = []
    connector = SimpleNamespace(
        cache_dsa_kv_topology=lambda value: received.append(value)
    )
    engine = AscendLMCacheEngine.__new__(AscendLMCacheEngine)
    engine.config = SimpleNamespace(dsa_two_groups=True)
    engine.metadata = SimpleNamespace(runtime_kv_group_layer_counts=(79, 22))
    engine.gpu_connector = connector
    engine.dsa_kv_topology = None
    engine._dsa_kv_topology_view = None

    engine.cache_dsa_kv_topology(topology)

    assert engine.dsa_kv_topology is topology
    assert engine.metadata.dsa_kv_topology is topology
    assert received == [topology]


def test_topology_stage_does_not_change_transfer_or_non_dsa_state() -> None:
    topology = _topology()
    connector = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    connector.dsa_two_groups = True
    connector._group_layouts = {}
    connector.dsa_kv_topology = None
    connector._dsa_kv_topology_view = None
    events = (object(), object())
    streams = [object(), object()]
    connector._sparse_load_done_events = events
    connector.load_stream_list = streams

    connector.cache_dsa_kv_topology(topology)

    assert connector._sparse_load_done_events is events
    assert connector.load_stream_list is streams

    non_dsa = object.__new__(VLLMPagedMemLayerwiseNPUConnector)
    non_dsa.dsa_two_groups = False
    non_dsa.dsa_kv_topology = None
    non_dsa.cache_dsa_kv_topology(SimpleNamespace())
    assert non_dsa.dsa_kv_topology is None
