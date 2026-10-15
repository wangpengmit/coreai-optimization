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


def _choice(program):
    """The program's one ``param.if``."""
    (choice,) = [op for op in program._module._mlir_module.body.operations if op.name == "param.if"]
    return choice


def _cases(program, schemes) -> dict:
    """The graphs of each case of the program's ``param.if``, by name, keyed by scheme;
    ``None`` is the fp16 ``else`` case."""
    choice = _choice(program)
    keys = [*schemes, None]
    assert len(choice.regions) == len(keys)
    cases = {}
    for key, region in zip(keys, choice.regions, strict=True):
        (group,) = region.blocks[0].operations
        cases[key] = {
            op.attributes["sym_name"].value: op for op in group.regions[0].blocks[0].operations
        }
    return cases


def _scheme_attrs(program) -> dict[str, str]:
    """Each scheme's ``#coreai.weight_transform_scheme``, as text, by scheme name."""
    schemes = program._module._mlir_module.operation.attributes[SCHEMES_ATTR]
    texts = [str(attr) for attr in schemes]
    return {re.match(r'#coreai.weight_transform_scheme<"(\w+)"', t).group(1): t for t in texts}


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


def _run(module, pipeline: str) -> None:
    from coreai._compiler.passmanager import PassManager  # noqa: PLC0415

    with module.context:
        PassManager.parse(f"builtin.module({pipeline})").run(module.operation)


def _resolve(program, scheme: str | None) -> None:
    """Keep the case the compiler keeps for ``scheme`` (None: no scheme chosen)."""
    _run(
        program._module._mlir_module, f"resolve-properties{{compression-scheme={scheme or 'none'}}}"
    )


@pytest.fixture
def _universal(_coreai_program, tmp_path: Path):
    program, _, _ = _coreai_program
    return program, make_universal_asset(program, _SCHEMES, tmp_path)


def test_one_copy_of_each_weight(_universal) -> None:
    program, universal = _universal
    from coreai.authoring import AIModelAsset  # noqa: PLC0415

    loaded_program = AIModelAsset.load(universal.path).program
    loaded = _asm(loaded_program)
    resources = set(re.findall(r"resource_\d+", loaded))
    assert resources == set(re.findall(r"resource_\d+", _asm(program)))
    # Every case has the program's one graph, under its own name; no graph is outside them.
    assert {key: list(graphs) for key, graphs in _cases(loaded_program, _SCHEMES).items()} == {
        key: ["main"] for key in [*_SCHEMES, None]
    }
    assert not [
        op
        for op in loaded_program._module._mlir_module.body.operations
        if op.name == "coreai.graph"
    ]
    for scheme in _SCHEMES:
        assert f'param.check_property "compression_scheme" in ["{scheme}"]' in loaded


def test_scheme_entry_per_virtual_constant(_universal) -> None:
    from coreai.authoring import AIModelAsset  # noqa: PLC0415

    _, universal = _universal
    saved = _scheme_attrs(universal.program)
    # The attribute survives the versioned asset unchanged.
    assert _scheme_attrs(AIModelAsset.load(universal.path).program) == saved
    assert list(saved) == list(_SCHEMES)
    cases = _cases(universal.program, _SCHEMES)
    for scheme in _SCHEMES:
        ir = universal.schemes[scheme].ir
        assert "graphs =" not in saved[scheme]
        assert re.findall(r"\bt(\d+) = #coreai.weight_transform<", saved[scheme]) == [
            str(t["id"]) for t in ir["transforms"]
        ]
        expected = {(t["id"], role) for t in ir["transforms"] for role in t["results"]}
        virtual = _virtual_constants(cases[scheme]["main"], scheme)
        actual = [(transform_id, role) for _, _, transform_id, role in virtual]
        assert ir["transforms"], scheme
        assert sorted(actual) == sorted(expected)
    # The fp16 case has no virtual resource.
    assert not any(_virtual_constants(cases[None]["main"], s) for s in _SCHEMES)
    assert "udml.placeholder" not in _asm(universal.program)


def test_derivations_in_resource_section(_universal) -> None:
    """The saved asset shows how each virtual resource is computed."""
    from coreai.authoring import AIModelAsset  # noqa: PLC0415

    _, universal = _universal
    loaded = _asm(AIModelAsset.load(universal.path).program)
    resources = loaded.split("\n{-#\n")[1]
    cases = _cases(universal.program, _SCHEMES)
    for scheme in _SCHEMES:
        for _, source, transform_id, role in _virtual_constants(cases[scheme]["main"], scheme):
            key = f"{source}.{scheme}.t{transform_id}.{role}"
            derivation = rf"#coreai.derived_resource<@{scheme}::@t{transform_id}\[{role}\]"
            pattern = rf'{re.escape(key)}: "{derivation}\({source} : tensor<'
            assert re.search(pattern, resources), key


@pytest.mark.parametrize("scheme", list(_SCHEMES))
def test_refilled_graph_is_direct_twin(_universal, scheme: str, tmp_path: Path) -> None:
    """The case the compiler keeps for the scheme, with its virtual constants filled with
    coreai-opt's own values, gives back the twin exactly."""
    from coreai.authoring import AIModelAsset  # noqa: PLC0415

    _, universal = _universal
    program = copy.deepcopy(universal.program)
    _resolve(program, scheme)
    assert "param." not in _asm(program)
    del program._module._mlir_module.operation.attributes[SCHEMES_ATTR]
    (graph,) = program._module._mlir_module.body.operations
    removed = universal.schemes[scheme].removed_values
    virtual = _virtual_constants(graph, scheme)
    assert virtual
    for op, _, transform_id, role in virtual:
        op.attributes["value"] = removed[(transform_id, role)]
    program.save_asset(tmp_path / "refilled.aimodel")

    refilled = _asm(AIModelAsset.load(tmp_path / "refilled.aimodel").program)
    twin = _asm(AIModelAsset.load(universal.twins[scheme]).program)
    assert refilled == twin


def test_no_scheme_keeps_the_fp16_program(_universal) -> None:
    program, universal = _universal
    resolved = copy.deepcopy(universal.program)
    _resolve(resolved, None)
    del resolved._module._mlir_module.operation.attributes[SCHEMES_ATTR]

    def masked(p) -> str:
        # An inline fp16 weight became a resource, so payloads and the resource section differ.
        return re.sub(r"dense(_resource)?<[^>]*>", "<payload>", _asm(p).split("\n{-#\n")[0])

    assert masked(resolved) == masked(program)


def test_a_case_can_have_any_condition(_coreai_program, tmp_path: Path) -> None:
    """A scheme can be chosen by the device, here by its preferred compute unit."""
    from coreai_torch import properties  # noqa: PLC0415

    program, _, _ = _coreai_program
    schemes = {s: _SCHEMES[s] for s in ("w8", "lut4")}
    universal = make_universal_asset(
        program,
        schemes,
        tmp_path,
        conditions={"w8": properties.when(preferred_compute_unit="neural-engine")},
    )
    text = _asm(universal.program)
    assert 'param.check_property "preferred_compute_unit" in ["neural-engine"]' in text
    assert 'param.check_property "compression_scheme" in ["lut4"]' in text
    resolved = copy.deepcopy(universal.program)
    _run(
        resolved._module._mlir_module,
        "resolve-properties{preferred-compute-unit=neural-engine compression-scheme=lut4}",
    )
    assert _virtual_constants(resolved._module._mlir_module.body.operations[0], "w8")


def test_a_program_with_conditions_is_rejected(_coreai_program, tmp_path: Path) -> None:
    """The function builds the param.if itself, so a program can't have one already."""
    program, _, _ = _coreai_program
    universal = make_universal_asset(program, {"w8": _SCHEMES["w8"]}, tmp_path / "first")
    with pytest.raises(ValueError, match="graphs only"):
        make_universal_asset(universal.program, {"w8": _SCHEMES["w8"]}, tmp_path / "second")


@pytest.mark.parametrize(
    ("schemes", "conditions", "error"),
    [
        ({"none": _SCHEMES["w8"]}, None, "not 'none'"),
        ({}, None, "at least one scheme"),
        ({"w8": _SCHEMES["w8"]}, {"lut4": None}, "unknown schemes"),
    ],
)
def test_invalid_arguments(_coreai_program, tmp_path: Path, schemes, conditions, error) -> None:
    program, _, _ = _coreai_program
    with pytest.raises(ValueError, match=error):
        make_universal_asset(program, schemes, tmp_path, conditions=conditions)


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
    """Each case holds every graph, an unchanged callee too; graphs share a weight's transform."""
    from coreai.authoring import AIModelAsset  # noqa: PLC0415

    program = _parse_program(_MULTI_FUNCTION, seed=0)
    universal = make_universal_asset(program, {scheme: _SCHEMES[scheme]}, tmp_path)
    loaded_program = AIModelAsset.load(universal.path).program
    cases = _cases(loaded_program, [scheme])
    assert {key: list(graphs) for key, graphs in cases.items()} == {
        scheme: ["helper", "main", "prefill"],
        None: ["helper", "main", "prefill"],
    }
    assert "coreai.invoke @helper(" in str(cases[scheme]["main"])
    # Both functions read the same weight with the same parameters: one transform.
    ids = [t["id"] for t in universal.schemes[scheme].ir["transforms"]]
    assert ids == [0], ids
    cases = _cases(universal.program, [scheme])
    keys = [
        sorted(f"{i}.{role}" for _, _, i, role in _virtual_constants(cases[scheme][name], scheme))
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


def test_changed_callee_in_its_case(tmp_path: Path) -> None:
    """A callee whose weight the scheme compresses has virtual constants in the scheme's case,
    and its caller there calls it."""
    program = _parse_program(_CHANGED_CALLEE, seed=1)
    universal = make_universal_asset(program, {"w8": _SCHEMES["w8"]}, tmp_path)
    cases = _cases(universal.program, ["w8"])
    assert "coreai.invoke @helper(" in str(cases["w8"]["main"])
    assert _virtual_constants(cases["w8"]["helper"], "w8")
    assert not _virtual_constants(cases[None]["helper"], "w8")


def _materialized_constants(module) -> list:
    """The constant values of `module` once the compiler kept and materialized its w8 case."""
    from coreai_opt.coreai_utils._coreai_imports import (  # noqa: PLC0415
        _get_constant_value_as_np_array,
    )

    _run(
        module,
        "resolve-properties{compression-scheme=w8}, coreai-materialize-weight-transforms",
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
