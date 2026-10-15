# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Universal assets: N compressed graphs that share one fp16 copy of the weights.

Prototype 3 of the Universal Asset project (virtual resources). For each scheme,
the existing coreai-opt function rewrites a copy of the program ahead of time.
The weight constants it computed then become virtual resources: each keeps its
place, and its value is a ``dense_resource<<source>.<scheme>.t<id>.<role>>`` without
data, whose ``#coreai.derived_resource`` (in the ``coreai`` resource group)
computes it from the fp16 source resource. A ``#coreai.weight_transform_scheme``
attribute per scheme (in the module attribute ``coreai.weight_transform_schemes``)
holds the transforms. The compiler's ``coreai-materialize-weight-transforms``
pass computes the data at JIT compile time.

Prototype 5 (performance) moves the expensive fits ahead of time. coreai-opt's
k-means centers (the LUT, with its quantized scale and zero point) stay ordinary
constants with the twin's values, and the sparsity percentile becomes a threshold.
The transform's ``aot`` dictionary holds what the JIT needs to recompute the
other results exactly: per block, the cluster ``bounds`` and ``labels``, or the
magnitude ``threshold``.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
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
# Results of the fits that coreai-opt does ahead of time (Prototype 5): they stay
# ordinary constants with the twin's values.
_AOT_REAL_ROLES = {
    "palettize": frozenset({"lut", "scale", "zero_point"}),
    "sparsify": frozenset({"lut"}),
}


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
    graphs = {entry: "@" + _mlir_string(graph) for entry, graph in ir["graphs"].items()}
    options = {k: _mlir_attr(v) for k, v in ir["options"].items() if v is not None}
    transforms = {
        f"t{t['id']}": f"#coreai.weight_transform<{t['kernel']}, {_params_attr(t['params'])}>"
        for t in ir["transforms"]
    }
    return (
        f"#coreai.weight_transform_scheme<{_mlir_string(ir['scheme'])}, "
        f"graphs = {_mlir_dict(graphs)}, options = {_mlir_dict(options)}, "
        f"transforms = {_mlir_dict(transforms)}>"
    )


@dataclass
class DeferredScheme:
    """What ``defer_weights`` produced for one scheme."""

    ir: dict[str, Any]
    # Value attribute of every constant that became a virtual resource, keyed
    # by (transform id, role). Lets tests rebuild the direct twin exactly.
    removed_values: dict[tuple[int, str], Any] = field(default_factory=dict)
    # Transform id of each distinct (source resource, source type, parameters, AOT
    # digest). The digest separates two copies of a weight whose twins differ
    # (scikit-learn's k-means can cluster two copies differently).
    transform_ids: dict[tuple[str, str, str, str], int] = field(default_factory=dict)
    # The `aot` dictionary of each transform that has one, by transform id.
    aot: dict[int, Any] = field(default_factory=dict)
    # Bytes of `aot` data, and the chains that keep Prototype 3's form (every
    # result virtual): their AOT check failed, or a result was a dropped splat.
    aot_bytes: int = 0
    aot_fallbacks: list[str] = field(default_factory=list)


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


def _constant_values(op: Any) -> np.ndarray:
    """A constant's values as a numpy array; integers of any width are unpacked."""
    from coreai._compiler._mlir_libs._coreaiIR._bindings import compiler  # noqa: PLC0415

    tensor = op.result.type
    shape = list(tensor.shape)
    count = math.prod(shape)
    element = str(tensor.element_type)
    width = tensor.element_type.width
    raw = np.zeros(math.ceil(count * max(width, 8) / 8) + 8, dtype=np.uint8)
    compiler.copy_constant_value_to_array(constant_op=op, destination=raw)
    if element in ("f16", "f32"):
        dtype = np.float16 if element == "f16" else np.float32
        return raw[: count * dtype().itemsize].view(dtype).reshape(shape)
    if element == "i1":  # A bool: one byte per element.
        return raw[:count].reshape(shape)
    signed = not element.startswith("ui")
    if width >= 8:
        dtype = np.dtype(f"{'i' if signed else 'u'}{width // 8}")
        return raw[: count * dtype.itemsize].view(dtype).reshape(shape)
    packed = raw[: (count * width + 7) // 8]
    if 8 % width == 0:  # LSB first, `8 // width` values per byte.
        mask = (1 << width) - 1
        parts = [(packed >> shift) & mask for shift in range(0, 8, width)]
        values = np.stack(parts, axis=-1).ravel()[:count]
    else:
        bits = np.unpackbits(packed, bitorder="little")[: count * width].reshape(count, width)
        values = (bits.astype(np.uint8) << np.arange(width, dtype=np.uint8)).sum(1, dtype=np.uint8)
    if signed:
        values = values.astype(np.int8)
        values = np.where(values >= 1 << (width - 1), values - (1 << width), values).astype(np.int8)
    return values.reshape(shape)


def _interval_aot(
    values: np.ndarray, indices: np.ndarray, n_bits: int
) -> dict[str, np.ndarray] | None:
    """``bounds`` and ``labels`` such that ``labels[g, #(bounds[g] < v)]`` is the index of
    each value ``v`` of block ``g`` (a row of ``values`` and ``indices``).

    In 1-D, each k-means cluster is an interval of its block's values, for both of
    coreai-opt's algorithms (optimal DP and scikit-learn). ``bounds[g]`` holds the
    largest value of each cluster but the last, in ascending order, padded with
    +inf, in the values' type. ``labels[g]`` holds the cluster of each interval; it
    is left out when it is the identity in every block. Returns None if a value is
    NaN or a cluster is not an interval.
    """
    k = 1 << n_bits
    blocks, size = values.shape
    # Column chunks of about 16M elements bound the temporaries.
    step = max(1, (1 << 24) // max(blocks, 1))
    chunks = [slice(c, c + step) for c in range(0, size, step)]
    low = np.full((blocks, k), np.inf, np.float32)
    high = np.full((blocks, k), -np.inf, np.float32)
    present = np.zeros((blocks, k), bool)
    for chunk in chunks:
        wide = values[:, chunk].astype(np.float32)  # Exact for fp16 and fp32.
        if np.isnan(wide).any():
            return None
        part = indices[:, chunk]
        for j in range(k):
            selected = part == j
            present[:, j] |= selected.any(axis=1)
            low[:, j] = np.minimum(
                low[:, j], np.where(selected, wide, np.inf).min(axis=1, initial=np.inf)
            )
            high[:, j] = np.maximum(
                high[:, j], np.where(selected, wide, -np.inf).max(axis=1, initial=-np.inf)
            )
    # Present clusters first, by their smallest value.
    order = np.lexsort((np.where(present, low, 0), ~present), axis=1)
    low, high, present = (np.take_along_axis(a, order, 1) for a in (low, high, present))
    if not ((~present[:, 1:]) | (low[:, 1:] > high[:, :-1])).all():
        return None
    bounds = np.where(present[:, 1:], high[:, :-1], np.inf).astype(values.dtype)
    labels = order.astype(np.uint8)
    # The JIT rule, in numpy: count the bounds below each value (NaN counts none).
    wide_bounds = bounds.astype(np.float32)
    for chunk in chunks:
        wide = values[:, chunk].astype(np.float32)
        counts = np.zeros(wide.shape, np.uint8)
        for i in range(k - 1):
            counts += wide > wide_bounds[:, i : i + 1]
        if not np.array_equal(
            np.take_along_axis(labels, counts.astype(np.intp), 1), indices[:, chunk]
        ):
            return None
    aot = {"bounds": bounds}
    if not ((labels == np.arange(k)) | ~present).all():
        aot["labels"] = labels
    return aot


def _palettize_aot(source: np.ndarray, params: dict[str, Any], indices: np.ndarray) -> dict | None:
    """``palettize_weights``' preprocessing, then the cluster bounds of each channel group."""
    weight = source
    axis = params["channel_axis"]
    blocked = [d for d, b in enumerate(params["block_sizes"]) if b not in (0, weight.shape[d])]
    if params["cluster_dim"] != 1 or set(blocked) - {axis} or indices.shape != weight.shape:
        return None
    if params["enable_per_channel_scale"]:
        scale = np.max(np.abs(weight), axis=axis, keepdims=True)
        scale[scale == 0] = 1
        weight = weight / scale
    weight = weight.astype(np.float16)
    if params["enable_fast_kmeans_mode"]:
        weight = np.round(weight, decimals=params["rounding_precision"])
    channels = weight.shape[axis]
    group = params["block_sizes"][axis] or channels
    blocks = channels // group
    grouped = np.moveaxis(weight, axis, 0).reshape(blocks, -1)
    grouped_indices = np.moveaxis(indices, axis, 0).reshape(blocks, -1)
    return _interval_aot(grouped, grouped_indices, params["n_bits"])


def _magnitude_threshold(source: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """The largest magnitude that the twin pruned (-inf if none): the JIT prunes a
    weight if ``|w| <= threshold``, and keeps it if it is also non-zero."""
    kept = mask.astype(bool)
    pruned = ~kept & (source != 0)
    magnitudes = np.abs(source)
    threshold = magnitudes[pruned].max() if pruned.any() else np.array(-np.inf, source.dtype)
    if not np.array_equal(~(magnitudes <= threshold) & (source != 0), kept):
        return None
    return np.array([threshold], dtype=source.dtype)


def _chain_aot(
    kernel: str, params: dict[str, Any], source: Any, constants: list[Any]
) -> dict | None:
    """The `aot` data for one weight's twin, or None if the JIT needs Prototype 3's form."""
    if kernel not in _AOT_REAL_ROLES:
        return None
    roles = {_role(op): op for op in constants}
    weight = _get_constant_value_as_np_array(source)

    def indices_of(shape: tuple[int, ...]) -> np.ndarray:
        # A splat-zero `indices` is not among the constants: every value is in cluster 0.
        if "indices" not in roles:
            return np.zeros(shape, np.uint8)
        return _constant_values(roles["indices"]).reshape(shape)

    if kernel == "palettize":
        return _palettize_aot(weight, params, indices_of(weight.shape))
    sparsity = params["target_sparsity"]
    if (
        sparsity is None
        or not 0 < sparsity < 1
        or params["block_size"] is not None
        or "mask" not in roles
    ):
        return None
    mask = _constant_values(roles["mask"])
    threshold = _magnitude_threshold(weight, mask)
    if threshold is None:
        return None
    aot = {"threshold": threshold}
    nonzero = weight[mask.astype(bool)].reshape(1, -1)
    if params["palettize_nbits"] is not None and nonzero.size:
        intervals = _interval_aot(nonzero, indices_of(nonzero.shape), params["palettize_nbits"])
        if intervals is None:
            return None
        aot.update(intervals)
    return aot


def _element_type(dtype: np.dtype) -> Any:
    from coreai._compiler.ir import F16Type, F32Type, IntegerType  # noqa: PLC0415

    if dtype == np.float16:
        return F16Type.get()
    if dtype == np.float32:
        return F32Type.get()
    if dtype == np.uint8:
        return IntegerType.get_unsigned(8)
    raise TypeError(f"no element type for {dtype}")


def _aot_attr(aot: dict[str, np.ndarray], key_prefix: str) -> Any:
    """The ``aot`` dictionary attribute: a resource per array (an inline ``dense``
    for a single value). Must run inside the program's context."""
    from coreai._compiler.ir import DictAttr, RankedTensorType  # noqa: PLC0415

    entries = {}
    for name, array in aot.items():
        array = np.ascontiguousarray(array)
        tensor = RankedTensorType.get(list(array.shape), _element_type(array.dtype))
        if array.size == 1 and array.dtype.kind == "f":
            entries[name] = _DenseElementsAttr.get(array)
        else:
            entries[name] = _DenseResourceElementsAttr.get_from_buffer(
                array, f"{key_prefix}.{name}", tensor
            )
    return DictAttr.get(entries)


def defer_weights(
    graph: Any,
    eligible: list[tuple[Any, dict[str, Any]]],
    preexisting: set[Any],
    kernel: str,
    deferred: DeferredScheme,
    source_keys: dict[Any, str],
    *,
    aot: bool = True,
) -> None:
    """Turn the weight-derived constants coreai-opt created in ``graph`` into virtual resources.

    coreai-opt inserts each weight's compression chain right before the source
    constant and leaves the source dead. For each such source, every new leaf
    constant of its chain keeps its position and location, but its value becomes
    ``dense_resource<<source>.<scheme>.t<id>.<role>>`` without data, which a
    ``#coreai.derived_resource`` computes from the source resource. Structural
    constants and splat zeros stay, and the dead source is erased.
    ``source_keys`` maps each source op to its fp16 resource. A transform id
    covers one distinct (source resource, source type, parameters, AOT data) of
    the scheme, so graphs that share a weight share its virtual resources. A
    resource key has one meaning in a context; the source key is a content
    hash, so the virtual keys of two models do not collide.

    With ``aot``, the fits of palettization and magnitude sparsity are done here:
    the LUT (with its scale and zero point) keeps the twin's value, and the
    transform's ``aot`` dictionary lets the JIT recompute the other results bit for
    bit. A weight whose AOT check fails keeps Prototype 3's form.
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
        chain_aot = _chain_aot(kernel, params, source, constants) if aot else None
        real_roles = (
            _AOT_REAL_ROLES.get(kernel, frozenset()) if chain_aot is not None else frozenset()
        )
        digest = ""
        if chain_aot is not None:
            # The virtual results depend only on the source, the parameters and the
            # AOT data; the fit results stay constants of each copy.
            hasher = hashlib.blake2b(digest_size=12)
            for name in sorted(chain_aot):
                hasher.update(name.encode() + chain_aot[name].tobytes())
            digest = hasher.hexdigest()
        elif aot and kernel in _AOT_REAL_ROLES and _chain_aot_expected(kernel, params):
            deferred.aot_fallbacks.append(f"{source_key} {params}")
        transform_key = (
            source_key,
            str(source.result.type),
            json.dumps(params, sort_keys=True),
            digest,
        )
        transform_id = deferred.transform_ids.get(transform_key)
        if transform_id is None:
            transform_id = len(deferred.ir["transforms"])
            deferred.transform_ids[transform_key] = transform_id
            deferred.ir["transforms"].append(
                {"id": transform_id, "kernel": kernel, "params": params, "results": []}
            )
            if chain_aot is not None:
                with source.context, source.location:
                    deferred.aot[transform_id] = _aot_attr(
                        chain_aot, f"{source_key}.{deferred.ir['scheme']}.t{transform_id}"
                    )
                deferred.aot_bytes += sum(a.nbytes for a in chain_aot.values())
        results = deferred.ir["transforms"][transform_id]["results"]
        for op in constants:
            role = _role(op)
            if role in real_roles:
                continue  # The fit's result: an ordinary constant with the twin's value.
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


def _chain_aot_expected(kernel: str, params: dict[str, Any]) -> bool:
    """Whether a transform of these parameters should get ``aot`` data."""
    if kernel == "palettize":
        return params["cluster_dim"] == 1
    sparsity = params["target_sparsity"]
    return sparsity is not None and 0 < sparsity < 1 and params["block_size"] is None


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


_SYMBOL_REF = re.compile(r'@"?([\w.$-]+)"?')


def _callees(graph: Any) -> set[str]:
    """Symbols that ``graph``'s ops reference, also inside other attributes
    (``coreai.invoke`` wraps its callee in ``#param.constant<@callee>``)."""
    callees = set()
    for op in _iter_ops(graph):
        attributes = op.attributes
        for i in range(len(attributes)):
            callees.update(_SYMBOL_REF.findall(str(attributes[i].attr)))
    return callees


def _is_public(graph: Any) -> bool:
    return "priv" not in graph.attributes


def make_universal_asset(
    program: _AIProgram,
    schemes: dict[str, Scheme],
    out_dir: Path,
    *,
    aot: bool = True,
) -> UniversalAsset:
    """Write ``out_dir/universal.aimodel`` and each scheme's direct twin.

    ``universal.aimodel`` holds the fp16 program plus, per scheme, a
    ``<graph>__<scheme>`` copy of every public graph and of every callee that
    the scheme changes, and a ``#coreai.weight_transform_scheme`` in the module
    attribute ``coreai.weight_transform_schemes``.
    ``out_dir/twins/<scheme>.aimodel`` is coreai-opt's direct output for the
    scheme, used for comparison. ``aot=False`` gives Prototype 3's asset: no fit
    is done ahead of time.
    """
    import gc  # noqa: PLC0415

    from coreai._compiler.ir import ArrayAttr, Attribute, StringAttr, SymbolTable  # noqa: PLC0415
    from coreai.authoring.asset import OSVersion  # noqa: PLC0415

    names = [_name(g) for g in _graphs(program)]
    entries = [_name(g) for g in _graphs(program) if _is_public(g)]
    if not entries:
        raise ValueError("expected a program with a public graph")
    for name in names:
        # The pass treats `<graph>__<anything>` as a scheme graph, at any `__`.
        if any(name[:i] in names for i in range(1, len(name)) if name.startswith("__", i)):
            raise ValueError(f"graph name {name!r} looks like a scheme graph")

    universal = copy.deepcopy(program)
    fp16_graphs = {_name(g): g for g in _graphs(universal)}
    inline_resources: dict[Any, Any] = {}
    universal_body_end = _graphs(universal)[-1]
    twins: dict[str, Path] = {}
    deferred_schemes: dict[str, DeferredScheme] = {}
    for scheme, (fn, kwargs) in schemes.items():
        if not re.fullmatch(r"[A-Za-z0-9_]+", scheme):
            raise ValueError(f"scheme name must be alphanumeric/underscore, got {scheme!r}")
        if fn not in _KERNELS:
            raise ValueError(f"unsupported compression function {fn!r}")
        if "in_place" in kwargs or "coreai_program" in kwargs:
            raise ValueError("scheme kwargs must not set coreai_program or in_place")
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

        # Callees whose weights the scheme left alone stay shared with the fp16 program.
        changed = {
            name
            for name, sources in eligible.items()
            if any(not list(source.result.uses) for source, _ in sources)
        }
        moved = [name for name in graphs if name in entries or name in changed]
        for name, graph in graphs.items():
            if name not in moved and _callees(graph) & (set(moved) - set(entries)):
                raise NotImplementedError(
                    f"unchanged graph {name!r} calls a callee that {scheme!r} changes"
                )
        deferred = DeferredScheme(
            ir={
                "scheme": scheme,
                "graphs": {name: f"{name}__{scheme}" for name in moved},
                "options": kw,
                "transforms": [],
            }
        )
        for name in moved:
            defer_weights(
                graphs[name],
                eligible[name],
                preexisting,
                kernel.name,
                deferred,
                source_keys,
                aot=aot,
            )
        for name in moved:
            graph = graphs[name]
            with graph.context:
                graph.attributes["sym_name"] = StringAttr.get(f"{name}__{scheme}")
                for other in moved:
                    SymbolTable.replace_all_symbol_uses(
                        name, f"{name}__{scheme}", graphs[other].operation
                    )
        for name in moved:
            graphs[name].move_after(universal_body_end)
            universal_body_end = graphs[name]
        deferred_schemes[scheme] = deferred
        del work, graphs, preexisting, eligible
        gc.collect()

    path = out_dir / "universal.aimodel"
    module = universal._module._mlir_module  # noqa: SLF001
    from coreai._compiler._mlir_libs._coreaiIR._bindings import compiler  # noqa: PLC0415
    from coreai._compiler.ir import DictAttr  # noqa: PLC0415

    with module.context:
        scheme_attrs = []
        for d in deferred_schemes.values():
            attr = Attribute.parse(_scheme_attr(d.ir))
            if d.aot:
                attr = compiler.set_weight_transform_aot(
                    scheme=attr, aot=DictAttr.get({f"t{i}": a for i, a in d.aot.items()})
                )
            scheme_attrs.append(attr)
        module.operation.attributes[SCHEMES_ATTR] = ArrayAttr.get(scheme_attrs)
    # The weight-transform attributes are experimental in the versioned dialect.
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
