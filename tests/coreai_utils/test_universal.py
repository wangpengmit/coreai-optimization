# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import copy
import json
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
from coreai_opt.coreai_utils.universal import (
    PLACEHOLDER_OPNAME,
    RESULT_ATTR,
    TRANSFORM_ID_ATTR,
    WEIGHT_TRANSFORMS_DIR,
    make_universal_asset,
)

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


def _placeholders(graph):
    return [
        op
        for op in graph.regions[0].blocks[0].operations
        if op.name == "udml.placeholder" and op.attributes["opname"].value == PLACEHOLDER_OPNAME
    ]


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


def test_json_entry_per_placeholder(_universal) -> None:
    _, universal = _universal
    for scheme in _SCHEMES:
        ir = json.loads((universal.path / WEIGHT_TRANSFORMS_DIR / f"{scheme}.json").read_text())
        assert ir["graphs"] == {"main": f"main__{scheme}"}
        expected = {(t["id"], role) for t in ir["transforms"] for role in t["results"]}
        placeholders = _placeholders(_graph(universal.program, f"main__{scheme}"))
        actual = [
            (op.attributes[TRANSFORM_ID_ATTR].value, op.attributes[RESULT_ATTR].value)
            for op in placeholders
        ]
        assert ir["transforms"], scheme
        assert sorted(actual) == sorted(expected)


@pytest.mark.parametrize("scheme", list(_SCHEMES))
def test_refilled_graph_is_direct_twin(_universal, scheme: str, tmp_path: Path) -> None:
    """Filling the placeholders with coreai-opt's own values gives back the twin exactly."""
    from coreai._compiler.dialects import coreai  # noqa: PLC0415
    from coreai._compiler.ir import InsertionPoint, StringAttr  # noqa: PLC0415
    from coreai.authoring import AIModelAsset  # noqa: PLC0415

    _, universal = _universal
    program = copy.deepcopy(universal.program)
    graph = _graph(program, f"main__{scheme}")
    for op in list(program._module._mlir_module.body.operations):
        if op != graph:
            op.erase()
    removed = universal.schemes[scheme].removed_values
    sources = set()
    for placeholder in _placeholders(graph):
        key = (
            placeholder.attributes[TRANSFORM_ID_ATTR].value,
            placeholder.attributes[RESULT_ATTR].value,
        )
        sources.add(placeholder.operands[0].owner)
        with placeholder.context, placeholder.location, InsertionPoint(placeholder):
            constant = coreai.ConstantOp(value=removed[key])
        placeholder.results[0].replace_all_uses_with(constant.result)
        placeholder.erase()
    for source in sources:
        assert not list(source.result.uses)
        source.erase()
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
    ir = json.loads((universal.path / WEIGHT_TRANSFORMS_DIR / f"{scheme}.json").read_text())
    assert ir["graphs"] == {"main": f"main__{scheme}", "prefill": f"prefill__{scheme}"}
    ids = [t["id"] for t in ir["transforms"]]
    assert ids == list(range(2)), ids
    for name in ("main", "prefill"):
        graph = _graph(universal.program, f"{name}__{scheme}")
        assert _placeholders(graph)


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
    ir = json.loads((universal.path / WEIGHT_TRANSFORMS_DIR / "w8.json").read_text())
    assert ir["graphs"] == {"helper": "helper__w8", "main": "main__w8"}
    main_text = re.search(r"coreai\.graph @main__w8\(.*?\n}", loaded, re.S).group(0)
    assert "@helper__w8" in main_text and "@helper(" not in main_text
    assert _placeholders(_graph(universal.program, "helper__w8"))
