# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Universal assets: N compressed graphs that share one fp16 copy of the weights.

Prototype 4 of the Universal Asset project (virtual resources, chosen by
conditions). For each scheme, the existing coreai-opt function rewrites a copy
of the program ahead of time. The weight constants it computed then become
virtual resources: each keeps its place, and its value is a
``dense_resource<<source>.<scheme>.t<id>.<role>>`` without data, whose
``#coreai.derived_resource`` (in the ``coreai`` resource group) computes it from
the fp16 source resource. A ``#coreai.weight_transform_scheme`` attribute per
scheme (in the module attribute ``coreai.weight_transform_schemes``) holds the
transforms.

A ``param.if`` (coreai-torch's ``properties.select``) chooses the scheme: one
case per scheme, by default when the app's ``compression_scheme`` property is
the scheme's name, and an ``else`` case with the fp16 program. The compiler's
``resolve-properties`` keeps one case, and its
``coreai-materialize-weight-transforms`` pass then computes the data of the
virtual resources at JIT compile time.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import re
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from coreai_opt.coreai_utils._coreai_imports import (
    AIProgram as _AIProgram,
    DenseElementsAttr as _DenseElementsAttr,
    DenseResourceElementsAttr as _DenseResourceElementsAttr,
    _get_constant_value_as_np_array,
)
from coreai_opt.coreai_utils._utils.graph_utils import (
    _infer_quantization_block_sizes,
    _select_input_output_channel_axis,
    _should_compress_op,
)
from coreai_opt.coreai_utils._utils.palettize_utils import (
    _infer_palettization_block_sizes_and_channel_axis,
)
from coreai_opt.coreai_utils.common import DType, QScheme
from coreai_opt.coreai_utils.passes import _OPS_WEIGHT_NEED_COMPRESSION
from coreai_opt.coreai_utils.passes.weight_palettization import palettize_weights
from coreai_opt.coreai_utils.passes.weight_quantization import quantize_weights
from coreai_opt.coreai_utils.passes.weight_sparsification import sparsify_weights

SCHEMES_ATTR = "coreai.weight_transform_schemes"

# The element type that a weight-transform parameter names for each `DType`.
_DTYPE_TYPES = {
    DType.INT2: "si2",
    DType.UINT2: "ui2",
    DType.INT4: "si4",
    DType.UINT4: "ui4",
    DType.INT8: "si8",
    DType.UINT8: "ui8",
    DType.FP4_E2M1FN: "f4E2M1FN",
    DType.FP8_E4M3FN: "f8E4M3FN",
    DType.FP8_E5M2: "f8E5M2",
    DType.FP8_E8M0FNU: "f8E8M0FNU",
}
_DTYPE_PARAMS = frozenset({"dtype", "scale_dtype", "lut_dtype", "quantize_dtype"})

# A scheme is a coreai-opt compression function plus its keyword arguments.
Scheme = tuple[Callable[..., _AIProgram], dict[str, Any]]

# Role of a deferred constant, keyed by the (consumer op, operand index) it feeds.
_ROLES = {
    ("coreai.blockwise_shift_scale", 0): "data",
    ("coreai.blockwise_shift_scale", 1): "scale",
    ("coreai.blockwise_shift_scale", 2): "zero_point",
    ("coreai.blockwise_shift_scale", 3): "offset2",
    ("coreai.lut_to_dense", 0): "indices",
    ("coreai.lut_to_dense", 1): "lut",
    ("coreai.lut_to_dense", 2): "axis",
    ("coreai.build_sparse_with_bitmask", 0): "nonzero_values",
    ("coreai.build_sparse_with_bitmask", 1): "mask",
}
# Structural constants that never depend on weight values.
_INLINE_ROLES = frozenset({"offset2", "axis"})


def _effective_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Bind ``kwargs`` to ``fn`` and fill in its defaults, so the IR is explicit."""
    bound = inspect.signature(fn).bind(coreai_program=None, **kwargs)
    bound.apply_defaults()
    resolved = dict(bound.arguments)
    resolved.pop("coreai_program")
    resolved.pop("in_place", None)
    return resolved


def _quantize_params(op: Any, kw: dict[str, Any]) -> dict[str, Any]:
    shape = list(op.result.type.shape)
    dtype = DType(kw["dtype"])
    scale_dtype = kw["scale_dtype"]
    if dtype == DType.FP4_E2M1FN:
        scale_dtype = DType.FP8_E8M0FNU  # quantize_weights always stores MXFP4 scales in e8m0.
    block_sizes = _infer_quantization_block_sizes(op, shape, kw["granularity"], kw["block_size"])
    return {
        "dtype": dtype.value,
        "qscheme": QScheme(kw["qscheme"]).value,
        "block_sizes": list(block_sizes),
        "scale_dtype": None if scale_dtype is None else DType(scale_dtype).value,
    }


def _palettize_params(op: Any, kw: dict[str, Any]) -> dict[str, Any]:
    shape = list(op.result.type.shape)
    block_sizes, channel_axis = _infer_palettization_block_sizes_and_channel_axis(
        op, shape, kw["granularity"], kw["group_size"]
    )
    lut_dtype = kw["lut_dtype"]
    return {
        "n_bits": kw["n_bits"],
        "lut_dtype": None if lut_dtype is None else DType(lut_dtype).value,
        "block_sizes": list(block_sizes),
        "channel_axis": channel_axis,
        "cluster_dim": kw["cluster_dim"],
        "enable_per_channel_scale": kw["enable_per_channel_scale"],
        "enable_fast_kmeans_mode": kw["enable_fast_kmeans_mode"],
        "rounding_precision": kw["rounding_precision"],
    }


def _percentile_index_dtype() -> str:
    """How this numpy computes np.percentile's virtual index: in the array's dtype
    (numpy < 2.4; fp16 then overflows above 65504 elements) or in float64."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        probe = np.percentile(np.ones(70000, dtype=np.float16), 50)
    return "weight" if np.isnan(probe) else "float64"


def _sparsify_params(op: Any, kw: dict[str, Any]) -> dict[str, Any]:
    input_axis, output_axis = _select_input_output_channel_axis(op)
    quantize_dtype = kw["quantize_dtype"]
    n_m_ratio = kw["n_m_ratio"]
    return {
        "target_sparsity": kw["target_sparsity"],
        "n_m_ratio": None if n_m_ratio is None else list(n_m_ratio),
        "block_size": kw["block_size"],
        "input_channel_axis": 1 if input_axis is None else input_axis,
        "output_channel_axis": 0 if output_axis is None else output_axis,
        "quantize_dtype": None if quantize_dtype is None else DType(quantize_dtype).value,
        "palettize_nbits": kw["palettize_nbits"],
        "percentile_index_dtype": _percentile_index_dtype(),
    }


@dataclass(frozen=True)
class _Kernel:
    name: str
    params: Callable[[Any, dict[str, Any]], dict[str, Any]]


_KERNELS: dict[Callable[..., Any], _Kernel] = {
    quantize_weights: _Kernel("quantize", _quantize_params),
    palettize_weights: _Kernel("palettize", _palettize_params),
    sparsify_weights: _Kernel("sparsify", _sparsify_params),
}


def _mlir_string(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _mlir_attr(value: Any) -> str:
    """``value`` (a coreai-opt option or parameter) in MLIR attribute syntax."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):  # Includes coreai-opt's string enums.
        return _mlir_string(str(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.17e} : f64"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_mlir_attr(v) for v in value) + "]"
    raise TypeError(f"cannot write {value!r} as an attribute")


def _mlir_dict(entries: dict[str, str]) -> str:
    return "{" + ", ".join(f"{_mlir_string(k)} = {v}" for k, v in entries.items()) + "}"


def _params_attr(params: dict[str, Any]) -> str:
    """A transform's parameters: dtypes as element types, the qscheme and numpy's
    percentile behavior as flags, and no entry for None."""
    entries = {}
    for key, value in params.items():
        if value is None:
            continue
        if key in _DTYPE_PARAMS:
            entries[key] = _DTYPE_TYPES[DType(value)]
        elif key == "qscheme":
            entries["symmetric"] = _mlir_attr(QScheme(value) == QScheme.SYMMETRIC)
        elif key == "percentile_index_dtype":
            entries["percentile_index_in_weight_type"] = _mlir_attr(value == "weight")
        else:
            entries[key] = _mlir_attr(value)
    return _mlir_dict(entries)


def _scheme_attr(ir: dict[str, Any]) -> str:
    """The ``#coreai.weight_transform_scheme`` attribute of one scheme's IR."""
    options = {k: _mlir_attr(v) for k, v in ir["options"].items() if v is not None}
    transforms = {
        f"t{t['id']}": f"#coreai.weight_transform<{t['kernel']}, {_params_attr(t['params'])}>"
        for t in ir["transforms"]
    }
    return (
        f"#coreai.weight_transform_scheme<{_mlir_string(ir['scheme'])}, "
        f"options = {_mlir_dict(options)}, transforms = {_mlir_dict(transforms)}>"
    )


@dataclass
class DeferredScheme:
    """What ``defer_weights`` produced for one scheme."""

    ir: dict[str, Any]
    # Value attribute of every constant that became a virtual resource, keyed
    # by (transform id, role). Lets tests rebuild the direct twin exactly.
    removed_values: dict[tuple[int, str], Any] = field(default_factory=dict)
    # Transform id of each distinct (source resource, source type, parameters).
    transform_ids: dict[tuple[str, str, str], int] = field(default_factory=dict)


@dataclass
class UniversalAsset:
    path: Path
    twins: dict[str, Path]
    schemes: dict[str, DeferredScheme]
    program: _AIProgram  # The in-memory universal program, as saved to ``path``.


def _iter_ops(op: Any) -> Iterator[Any]:
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                yield child
                yield from _iter_ops(child)


def _graphs(program: _AIProgram) -> list[Any]:
    return list(program._module._mlir_module.body.operations)  # noqa: SLF001


def _is_splat_zero(attr: Any) -> bool:
    if not isinstance(attr, _DenseElementsAttr) or not attr.is_splat:
        return False
    return float(attr.get_splat_value().value) == 0.0


def _role(constant: Any) -> str:
    uses = list(constant.result.uses)
    if len(uses) != 1:
        raise ValueError(f"expected one use of a compression constant, got {len(uses)}")
    key = (uses[0].owner.name, uses[0].operand_number)
    if key not in _ROLES:
        raise ValueError(f"unknown compression constant role {key}")
    return _ROLES[key]


def defer_weights(
    graph: Any,
    eligible: list[tuple[Any, dict[str, Any]]],
    preexisting: set[Any],
    kernel: str,
    deferred: DeferredScheme,
    source_keys: dict[Any, str],
) -> None:
    """Turn the weight-derived constants coreai-opt created in ``graph`` into virtual resources.

    coreai-opt inserts each weight's compression chain right before the source
    constant and leaves the source dead. For each such source, every new leaf
    constant of its chain keeps its position and location, but its value becomes
    ``dense_resource<<source>.<scheme>.t<id>.<role>>`` without data, which a
    ``#coreai.derived_resource`` computes from the source resource. Structural
    constants and splat zeros stay, and the dead source is erased.
    ``source_keys`` maps each source op to its fp16 resource. A transform id
    covers one distinct (source resource, source type, parameters) of the
    scheme, so graphs that share a weight share its virtual resources. A
    resource key has one meaning in a context; the source key is a content
    hash, so the virtual keys of two models do not collide.
    """
    from coreai._compiler._mlir_libs._coreaiIR._bindings import compiler  # noqa: PLC0415
    from coreai._compiler.ir import Attribute  # noqa: PLC0415

    scheme = deferred.ir["scheme"]
    # Find every chain first: a chain ends at the previous original op, so the
    # sources must stay until all chains are known.
    chains = []
    for source, params in eligible:
        if len(list(source.result.uses)) != 0:
            continue  # coreai-opt skipped this weight.
        block_ops = list(source.operation.block.operations)
        index = block_ops.index(source.operation)
        chain = []
        while index > 0 and block_ops[index - 1].operation not in preexisting:
            index -= 1
            chain.insert(0, block_ops[index])
        constants = [
            op
            for op in chain
            if op.name == "coreai.constant"
            and _role(op) not in _INLINE_ROLES
            and not _is_splat_zero(op.attributes["value"])
        ]
        if constants:
            chains.append((source, params, constants))

    for source, params, constants in chains:
        source_key = source_keys[source.operation]
        transform_key = (source_key, str(source.result.type), json.dumps(params, sort_keys=True))
        transform_id = deferred.transform_ids.get(transform_key)
        if transform_id is None:
            transform_id = len(deferred.ir["transforms"])
            deferred.transform_ids[transform_key] = transform_id
            deferred.ir["transforms"].append(
                {"id": transform_id, "kernel": kernel, "params": params, "results": []}
            )
        results = deferred.ir["transforms"][transform_id]["results"]
        for op in constants:
            role = _role(op)
            derivation = (
                f"#coreai.derived_resource<@{scheme}::@t{transform_id}[{role}]"
                f"({source_key} : {source.result.type})>"
            )
            deferred.removed_values[(transform_id, role)] = op.attributes["value"]
            with op.context, op.location:
                op.attributes["value"] = compiler.declare_virtual_resource(
                    type=op.result.type,
                    key=f"{source_key}.{scheme}.t{transform_id}.{role}",
                    derivation=Attribute.parse(derivation),
                )
            if role not in results:
                results.append(role)
    for source, _, _ in chains:
        source.operation.erase()


_RESOURCE_KEY = re.compile(r"dense_resource<([\w.$-]+)>")


def _source_resource(fp16_op: Any, inline_resources: dict[Any, Any]) -> str:
    """The resource key of an fp16 weight. A derivation names its source by key,
    so an inline weight becomes a resource with the same data. Its key is a
    content hash, like the keys of the other weights."""
    value = fp16_op.attributes["value"]
    match = _RESOURCE_KEY.match(str(value))
    if match:
        return match.group(1)
    if value not in inline_resources:
        data = np.ascontiguousarray(_get_constant_value_as_np_array(fp16_op))
        key = f"uasset_source_{hashlib.sha256(data.tobytes()).hexdigest()[:20]}"
        with fp16_op.context:
            inline_resources[value] = _DenseResourceElementsAttr.get_from_buffer(
                data, key, fp16_op.result.type
            )
    fp16_op.attributes["value"] = inline_resources[value]
    return _RESOURCE_KEY.match(str(inline_resources[value])).group(1)


def _name(graph: Any) -> str:
    return graph.attributes["sym_name"].value


def _is_public(graph: Any) -> bool:
    return "priv" not in graph.attributes


def make_universal_asset(
    program: _AIProgram,
    schemes: dict[str, Scheme],
    out_dir: Path,
    conditions: dict[str, Any] | None = None,
) -> UniversalAsset:
    """Write ``out_dir/universal.aimodel`` and each scheme's direct twin.

    ``universal.aimodel`` chooses its graphs with a ``param.if``: one case per
    scheme, in order, then an ``else`` case with the fp16 program. A scheme's
    case holds a copy of every graph of the program, under the same names,
    whose compressed weights are virtual resources; graphs share each fp16
    blob by key. The case's condition is ``conditions[scheme]``, a
    ``coreai_torch.properties.when(...)``; by default,
    ``when(compression_scheme=scheme)``. The module attribute
    ``coreai.weight_transform_schemes`` holds a ``#coreai.weight_transform_scheme``
    per scheme. ``out_dir/twins/<scheme>.aimodel`` is coreai-opt's direct output
    for the scheme, used for comparison.
    """
    import gc  # noqa: PLC0415

    from coreai._compiler.ir import ArrayAttr, Attribute, StringAttr, SymbolTable  # noqa: PLC0415
    from coreai.authoring.asset import OSVersion  # noqa: PLC0415
    from coreai_torch import properties  # noqa: PLC0415

    others = {op.operation.name for op in _graphs(program)} - {"coreai.graph"}
    if others:
        # This function builds the param.if itself, and select can't nest one.
        raise ValueError(f"expected a program of graphs only, got {sorted(others)}")
    names = [_name(g) for g in _graphs(program)]
    if not any(_is_public(g) for g in _graphs(program)):
        raise ValueError("expected a program with a public graph")
    if not schemes:
        raise ValueError("expected at least one scheme")
    conditions = dict(conditions or {})
    if set(conditions) - set(schemes):
        raise ValueError(f"conditions for unknown schemes {sorted(set(conditions) - set(schemes))}")

    universal = copy.deepcopy(program)
    fp16_graphs = {_name(g): g for g in _graphs(universal)}
    inline_resources: dict[Any, Any] = {}
    universal_body_end = _graphs(universal)[-1]
    twins: dict[str, Path] = {}
    deferred_schemes: dict[str, DeferredScheme] = {}
    for scheme, (fn, kwargs) in schemes.items():
        if not re.fullmatch(r"[A-Za-z0-9_]+", scheme) or scheme == "none":
            raise ValueError(
                f"scheme name must be alphanumeric/underscore and not 'none', got {scheme!r}"
            )
        if fn not in _KERNELS:
            raise ValueError(f"unsupported compression function {fn!r}")
        if "in_place" in kwargs or "coreai_program" in kwargs:
            raise ValueError("scheme kwargs must not set coreai_program or in_place")
        # Until select puts it in its case, each graph of the scheme has a name of its own.
        renamed = {name: f"{name}__{scheme}" for name in names}
        if set(renamed.values()) & set(names):
            raise ValueError(f"a graph name ends in '__{scheme}', the scheme's suffix")
        kernel = _KERNELS[fn]
        kw = _effective_kwargs(fn, kwargs)

        work = copy.deepcopy(program)
        graphs = {_name(g): g for g in _graphs(work)}
        preexisting = set()
        eligible: dict[str, list[tuple[Any, dict[str, Any]]]] = {name: [] for name in graphs}
        source_keys: dict[Any, str] = {}
        for name, graph in graphs.items():
            # The fp16 graph of the universal program holds each source weight.
            for op, fp16_op in zip(_iter_ops(graph), _iter_ops(fp16_graphs[name]), strict=True):
                preexisting.add(op.operation)
                threshold = kw["weight_num_threshold"]
                if _should_compress_op(op, threshold, _OPS_WEIGHT_NEED_COMPRESSION):
                    eligible[name].append((op, kernel.params(op, kw)))
                    source_keys[op.operation] = _source_resource(fp16_op, inline_resources)

        fn(coreai_program=work, in_place=True, **kwargs)
        twins[scheme] = out_dir / "twins" / f"{scheme}.aimodel"
        work.save_asset(twins[scheme])

        deferred = DeferredScheme(ir={"scheme": scheme, "options": kw, "transforms": []})
        for name, graph in graphs.items():
            defer_weights(graph, eligible[name], preexisting, kernel.name, deferred, source_keys)
        for name, graph in graphs.items():
            with graph.context:
                graph.attributes["sym_name"] = StringAttr.get(renamed[name])
                for other in graphs.values():
                    SymbolTable.replace_all_symbol_uses(name, renamed[name], other.operation)
        for graph in graphs.values():
            graph.move_after(universal_body_end)
            universal_body_end = graph
        deferred_schemes[scheme] = deferred
        del work, graphs, preexisting, eligible
        gc.collect()

    path = out_dir / "universal.aimodel"
    module = universal._module._mlir_module  # noqa: SLF001
    with module.context:
        module.operation.attributes[SCHEMES_ATTR] = ArrayAttr.get(
            [Attribute.parse(_scheme_attr(d.ir)) for d in deferred_schemes.values()]
        )
    cases = [
        (
            conditions.get(scheme) or properties.when(compression_scheme=scheme),
            [(f"{name}__{scheme}", name) for name in names],
        )
        for scheme in schemes
    ]
    properties.select(universal, [*cases, (None, names)])
    # The weight-transform attributes and the param ops are experimental in the
    # versioned dialect.
    universal.save_asset(path, minimum_os=OSVersion._experimental)  # noqa: SLF001
    return UniversalAsset(path=path, twins=twins, schemes=deferred_schemes, program=universal)


__all__ = [
    "SCHEMES_ATTR",
    "DeferredScheme",
    "Scheme",
    "UniversalAsset",
    "defer_weights",
    "make_universal_asset",
]
