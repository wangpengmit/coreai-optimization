# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import copy
import re
from pathlib import Path

import pytest

from coreai_opt.coreai_utils import (
    CompressionGranularity,
    DType,
    palettize_weights,
    quantize_weights,
    sparsify_weights,
)
from coreai_opt.coreai_utils.common import QScheme
from coreai_opt.coreai_utils.universal import SCHEMES_ATTR, make_universal_asset

_SCHEMES = {
    "w8": (quantize_weights, {"dtype": DType.INT8}),
    "w4_block": (
        quantize_weights,
        {"dtype": DType.INT4, "granularity": CompressionGranularity.PER_BLOCK},
    ),
    "uw8_asym": (quantize_weights, {"dtype": DType.UINT8, "qscheme": QScheme.ASYMMETRIC}),
    "fp8": (quantize_weights, {"dtype": DType.FP8_E4M3FN}),
    "lut4": (palettize_weights, {"n_bits": 4, "lut_dtype": None}),
    "lut4_int8_pcs": (
        palettize_weights,
        {
            "n_bits": 4,
            "lut_dtype": DType.INT8,
            "granularity": CompressionGranularity.PER_GROUPED_CHANNEL,
            "group_size": 16,
            "enable_per_channel_scale": True,
        },
    ),
    "s50": (sparsify_weights, {"target_sparsity": 0.5}),
    "s50_w8": (sparsify_weights, {"target_sparsity": 0.5, "quantize_dtype": DType.INT8}),
    "s50_lut4": (sparsify_weights, {"target_sparsity": 0.5, "palettize_nbits": 4}),
}


def _asm(program) -> str:
    return program._module._mlir_module.operation.get_asm(enable_debug_info=True)


def _graph(program, name: str):
    for op in program._module._mlir_module.body.operations:
        if op.attributes["sym_name"].value == name:
            return op
    raise KeyError(name)


def _scheme_attrs(program) -> dict[str, str]:
    """Each scheme's ``#coreai.weight_transform_scheme``, as text, by scheme name."""
    schemes = program._module._mlir_module.operation.attributes[SCHEMES_ATTR]
    texts = [str(attr) for attr in schemes]
    return {re.match(r'#coreai.weight_transform_scheme<"(\w+)"', t).group(1): t for t in texts}


def _scheme_graphs(text: str) -> dict[str, str]:
    graphs = re.search(r"graphs = \{(.*?)\}", text).group(1)
    return dict(re.findall(r"(\w+) = @(\w+)", graphs))


def _virtual_constants(graph, scheme: str):
    """(op, source key, transform id, role) of each constant whose value is a virtual resource."""
    pattern = re.compile(rf"dense_resource<(\w+)\.{scheme}\.t(\d+)\.(\w+)>")
    found = []
    for op in graph.regions[0].blocks[0].operations:
        if op.name == "coreai.constant":
            match = pattern.match(str(op.attributes["value"]))
            if match:
                found.append((op, match.group(1), int(match.group(2)), match.group(3)))
    return found


@pytest.fixture
def _universal(_coreai_program, tmp_path: Path):
    program, _, _ = _coreai_program
    return program, make_universal_asset(program, _SCHEMES, tmp_path)


def test_one_copy_of_each_weight(_universal) -> None:
    program, universal = _universal
    from coreai.authoring import AIModelAsset  # noqa: PLC0415

    loaded = _asm(AIModelAsset.load(universal.path).program)
    resources = set(re.findall(r"resource_\d+", loaded))
    assert resources == set(re.findall(r"resource_\d+", _asm(program)))
    graph_names = re.findall(r"coreai\.graph @(\w+)", loaded)
    assert graph_names == ["main"] + [f"main__{s}" for s in _SCHEMES]


def test_scheme_entry_per_placeholder(_universal) -> None:
    from coreai.authoring import AIModelAsset  # noqa: PLC0415

    _, universal = _universal
    saved = _scheme_attrs(universal.program)
    # The attribute survives the versioned asset unchanged.
    assert _scheme_attrs(AIModelAsset.load(universal.path).program) == saved
    assert list(saved) == list(_SCHEMES)
    for scheme in _SCHEMES:
        ir = universal.schemes[scheme].ir
        assert _scheme_graphs(saved[scheme]) == {"main": f"main__{scheme}"}
        assert re.findall(r"\bt(\d+) = #coreai.weight_transform<", saved[scheme]) == [
            str(t["id"]) for t in ir["transforms"]
        ]
        expected = {(t["id"], role) for t in ir["transforms"] for role in t["results"]}
        virtual = _virtual_constants(_graph(universal.program, f"main__{scheme}"), scheme)
        actual = [(transform_id, role) for _, _, transform_id, role in virtual]
        assert ir["transforms"], scheme
        assert sorted(actual) == sorted(expected)
        assert not any(op.name == "udml.placeholder" for op in _iter_graph_ops(universal.program))


def _iter_graph_ops(program):
    for graph in program._module._mlir_module.body.operations:
        yield from graph.regions[0].blocks[0].operations


def test_derivations_in_resource_section(_universal) -> None:
    """The saved asset shows how each virtual resource is computed."""
    from coreai.authoring import AIModelAsset  # noqa: PLC0415

    _, universal = _universal
    loaded = _asm(AIModelAsset.load(universal.path).program)
    resources = loaded.split("\n{-#\n")[1]
    for scheme in _SCHEMES:
        for _, source, transform_id, role in _virtual_constants(
            _graph(universal.program, f"main__{scheme}"), scheme
        ):
            key = f"{source}.{scheme}.t{transform_id}.{role}"
            derivation = rf"#coreai.derived_resource<@{scheme}::@t{transform_id}\[{role}\]"
            pattern = rf'{re.escape(key)}: "{derivation}\({source} : tensor<'
            assert re.search(pattern, resources), key


@pytest.mark.parametrize("scheme", list(_SCHEMES))
def test_refilled_graph_is_direct_twin(_universal, scheme: str, tmp_path: Path) -> None:
    """Filling the virtual constants with coreai-opt's own values gives back the twin exactly."""
    from coreai._compiler.ir import StringAttr  # noqa: PLC0415
    from coreai.authoring import AIModelAsset  # noqa: PLC0415

    _, universal = _universal
    program = copy.deepcopy(universal.program)
    graph = _graph(program, f"main__{scheme}")
    for op in list(program._module._mlir_module.body.operations):
        if op != graph:
            op.erase()
    del program._module._mlir_module.operation.attributes[SCHEMES_ATTR]
    removed = universal.schemes[scheme].removed_values
    virtual = _virtual_constants(graph, scheme)
    assert virtual
    for op, _, transform_id, role in virtual:
        op.attributes["value"] = removed[(transform_id, role)]
    with graph.context:
        graph.attributes["sym_name"] = StringAttr.get("main")
    program.save_asset(tmp_path / "refilled.aimodel")

    refilled = _asm(AIModelAsset.load(tmp_path / "refilled.aimodel").program)
    twin = _asm(AIModelAsset.load(universal.twins[scheme]).program)
    assert refilled == twin


_MULTI_FUNCTION = """
coreai.graph private noinline @helper(%x: tensor<1x64xf16>) -> tensor<1x64xf16> {
  %y = coreai.add %x, %x : (tensor<1x64xf16>, tensor<1x64xf16>) -> tensor<1x64xf16>
  coreai.output %y : tensor<1x64xf16>
}
coreai.graph @main(%x: tensor<1x32xf16>) -> tensor<1x64xf16> {
  %w = coreai.constant dense<WEIGHT> : tensor<32x64xf16>
  %y = coreai.batch_matmul %x, %w : (tensor<1x32xf16>, tensor<32x64xf16>) -> tensor<1x64xf16>
  %z = coreai.invoke @helper(%y) : (tensor<1x64xf16>) -> tensor<1x64xf16>
  coreai.output %z : tensor<1x64xf16>
}
coreai.graph @prefill(%x: tensor<3x32xf16>) -> tensor<3x64xf16> {
  %w = coreai.constant dense<WEIGHT> : tensor<32x64xf16>
  %y = coreai.batch_matmul %x, %w : (tensor<3x32xf16>, tensor<32x64xf16>) -> tensor<3x64xf16>
  coreai.output %y : tensor<3x64xf16>
}
"""


def _parse_program(text: str, seed: int):
    """An AIProgram from MLIR text whose `WEIGHT` is a seeded 32x64 fp16 literal."""
    import numpy as np  # noqa: PLC0415
    from coreai._compiler.ir import Module  # noqa: PLC0415
    from coreai.authoring.asset import AIProgram  # noqa: PLC0415
    from coreai.authoring.context import Context  # noqa: PLC0415

    weight = np.random.default_rng(seed).standard_normal((32, 64)).astype(np.float16)
    rows = ("[" + ", ".join(f"{v:.4f}" for v in row) + "]" for row in weight)
    literal = "[" + ", ".join(rows) + "]"
    module = Module.parse(text.replace("WEIGHT", literal), Context()._mlir_context)
    return AIProgram._from_mlir_module(module)


@pytest.mark.parametrize("scheme", ["w8", "lut4", "s50_w8"])
def test_multi_function_program(scheme: str, tmp_path: Path) -> None:
    """Every public graph gets a scheme copy; an unchanged callee stays shared."""
    from coreai.authoring import AIModelAsset  # noqa: PLC0415

    program = _parse_program(_MULTI_FUNCTION, seed=0)
    universal = make_universal_asset(program, {scheme: _SCHEMES[scheme]}, tmp_path)
    loaded = _asm(AIModelAsset.load(universal.path).program)
    graph_names = re.findall(r"coreai\.graph (?:private noinline )?@(\w+)", loaded)
    expected = ["helper", "main", "prefill", f"main__{scheme}", f"prefill__{scheme}"]
    assert graph_names == expected
    main_text = re.search(rf"coreai\.graph @main__{scheme}\(.*?\n}}", loaded, re.S).group(0)
    assert "coreai.invoke @helper(" in main_text
    schemes = _scheme_attrs(AIModelAsset.load(universal.path).program)
    assert _scheme_graphs(schemes[scheme]) == {
        "main": f"main__{scheme}",
        "prefill": f"prefill__{scheme}",
    }
    # Both functions read the same weight with the same parameters: one transform.
    ids = [t["id"] for t in universal.schemes[scheme].ir["transforms"]]
    assert ids == [0], ids
    keys = [
        sorted(
            f"{i}.{role}"
            for _, _, i, role in _virtual_constants(
                _graph(universal.program, f"{name}__{scheme}"), scheme
            )
        )
        for name in ("main", "prefill")
    ]
    assert keys[0] and keys[0] == keys[1]


_CHANGED_CALLEE = """
coreai.graph private noinline @helper(%x: tensor<1x32xf16>) -> tensor<1x64xf16> {
  %w = coreai.constant dense<WEIGHT> : tensor<32x64xf16>
  %y = coreai.batch_matmul %x, %w : (tensor<1x32xf16>, tensor<32x64xf16>) -> tensor<1x64xf16>
  coreai.output %y : tensor<1x64xf16>
}
coreai.graph @main(%x: tensor<1x32xf16>) -> tensor<1x64xf16> {
  %z = coreai.invoke @helper(%x) : (tensor<1x32xf16>) -> tensor<1x64xf16>
  coreai.output %z : tensor<1x64xf16>
}
"""


def test_changed_callee_gets_a_scheme_copy(tmp_path: Path) -> None:
    """A callee whose weight the scheme compresses is cloned, and its caller calls the clone."""
    from coreai.authoring import AIModelAsset  # noqa: PLC0415

    program = _parse_program(_CHANGED_CALLEE, seed=1)
    universal = make_universal_asset(program, {"w8": _SCHEMES["w8"]}, tmp_path)
    loaded = _asm(AIModelAsset.load(universal.path).program)
    graph_names = re.findall(r"coreai\.graph (?:private noinline )?@(\w+)", loaded)
    assert graph_names == ["helper", "main", "helper__w8", "main__w8"]
    schemes = _scheme_attrs(AIModelAsset.load(universal.path).program)
    assert _scheme_graphs(schemes["w8"]) == {"helper": "helper__w8", "main": "main__w8"}
    main_text = re.search(r"coreai\.graph @main__w8\(.*?\n}", loaded, re.S).group(0)
    assert "@helper__w8" in main_text and "@helper(" not in main_text
    assert _virtual_constants(_graph(universal.program, "helper__w8"), "w8")


def _materialized_constants(module) -> list:
    """The constant values of `module`'s w8 graph after materialization."""
    from coreai._compiler.passmanager import PassManager  # noqa: PLC0415

    from coreai_opt.coreai_utils._coreai_imports import (  # noqa: PLC0415
        _get_constant_value_as_np_array,
    )

    PassManager.parse("builtin.module(coreai-materialize-weight-transforms{scheme=w8})").run(
        module.operation
    )
    return [
        _get_constant_value_as_np_array(op)
        for graph in module.body.operations
        for op in graph.regions[0].blocks[0].operations
        if op.name == "coreai.constant"
    ]


def test_two_universal_assets_in_one_context(tmp_path: Path) -> None:
    """A resource key has one meaning in a context, so two models' virtual keys must differ."""
    import numpy as np  # noqa: PLC0415
    from coreai._compiler.context import Context  # noqa: PLC0415
    from coreai.authoring import AIModelAsset  # noqa: PLC0415

    paths = [
        make_universal_asset(
            _parse_program(_MULTI_FUNCTION, seed=seed), {"w8": _SCHEMES["w8"]}, tmp_path / str(seed)
        ).path
        for seed in (0, 1)
    ]
    alone = []
    for path in paths:
        with Context()._mlir_context:
            alone.append(
                _materialized_constants(AIModelAsset.load(path).program._module._mlir_module)
            )
    with Context()._mlir_context:
        modules = [AIModelAsset.load(path).program._module._mlir_module for path in paths]
        together = [_materialized_constants(module) for module in modules]

    assert not all(np.array_equal(a, b) for a, b in zip(*alone, strict=True))
    for expected, actual in zip(alone, together, strict=True):
        assert len(expected) == len(actual)
        for a, b in zip(expected, actual, strict=True):
            np.testing.assert_array_equal(a, b)
