"""Export a query-only initializer for CuTe's actual TVM-FFI library state.

The ordinary TVM wrapper lazily initializes a private CUDA library at first
invocation. An independent library load would duplicate it. Add an entry point
to the exported module that initializes that same state, with the original
runtime's once/cleanup semantics and without executing the tensor computation.
Only target-qualified prepared exports participate; eager compilation is intact.
"""

import ctypes
import hashlib
import importlib
from functools import wraps
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

_SOURCE = "54c569f7fefc014f09c54c460d7095c76d5ef327b2daf47fca2d7db1d90224ce"
_SYMBOL = "mlops_cuda_preload"
_lock = RLock()
_installed = False


def contract():
    module = importlib.import_module("cutlass.cutlass_dsl.tvm_ffi_provider")
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != _SOURCE:
        raise ValueError("unsupported CuTe TVM-FFI initialization contract")
    return {"version": 2, "source": _SOURCE}


def _with_initializer(module):
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import llvm

    with module.context:
        result = ir.Module.parse(str(module), context=module.context)
        states = [
            op.attributes["sym_name"].value
            for op in result.body.operations
            if op.operation.name == "llvm.mlir.global"
            and op.attributes["sym_name"].value.endswith("_cuda_state")
        ]
        if len(states) != 1:
            raise ValueError(
                "prepared CuTe object needs exactly one CUDA library state"
            )
        state = states[0]
        if not state.isidentifier():
            raise ValueError("unsupported CuTe CUDA state symbol")
        with ir.Location.unknown(), ir.InsertionPoint(result.body):
            initializer = llvm.LLVMFuncOp(
                _SYMBOL, ir.TypeAttr.get(ir.Type.parse("!llvm.func<i32 ()>"))
            )
            block = ir.Block.create_at_start(initializer.body, [])
            with ir.InsertionPoint(block):
                pointer = ir.Type.parse("!llvm.ptr")
                arguments = [
                    llvm.mlir_addressof(pointer, name)
                    for name in (
                        state,
                        "cuda_init",
                        "cuda_load_to_device",
                        "TVMFFIErrorSetRaisedFromCStr",
                    )
                ]
                status = llvm.call(
                    result=ir.IntegerType.get_signless(32),
                    callee="cuda_dialect_init_library_once",
                    callee_operands=arguments,
                    op_bundle_sizes=[],
                    op_bundle_operands=[],
                )
                llvm.return_(arg=status)
            release = llvm.LLVMFuncOp(
                "mlops_cuda_unload",
                ir.TypeAttr.get(ir.Type.parse("!llvm.func<void ()>")),
            )
            release_block = ir.Block.create_at_start(release.body, [])
            with ir.InsertionPoint(release_block):
                address = llvm.mlir_addressof(ir.Type.parse("!llvm.ptr"), state)
                llvm.call(
                    result=None,
                    callee="cuda_dialect_unload_library_once",
                    callee_operands=[address],
                    op_bundle_sizes=[],
                    op_bundle_operands=[],
                )
                llvm.return_()
        result.operation.verify()
        return result


def install():
    global _installed
    contract()
    with _lock:
        if _installed:
            return
        from cutlass.cutlass_dsl.tvm_ffi_provider import TVMFFIJitCompiledFunctionBase

        original = TVMFFIJitCompiledFunctionBase.export_to_c

        @wraps(original)
        def export(function, *args, **kwargs):
            from ._cache_scope import _active

            scope = _active.get()
            if scope is None or scope.directory is None:
                return original(function, *args, **kwargs)
            if kwargs.get("export_only_tvm_ffi_symbols", False):
                raise ValueError("prepared CuTe export must preserve its initializer")
            # The checked exporter reads only these fields. Clone the IR; do
            # not change the already compiled object or its live TVM function.
            prepared = SimpleNamespace(
                host_target=function.host_target,
                ir_module=_with_initializer(function.ir_module),
                function_name=function.function_name,
            )
            return original(prepared, *args, **kwargs)

        TVMFFIJitCompiledFunctionBase.export_to_c = export
        _installed = True


def initialize(module):
    """Load the shared CUDA state; never execute the GEMM or its epilogue."""
    contract()
    address = module.engine.lookup("__cute_internal_func_" + _SYMBOL)
    if not address:
        raise ValueError("prepared CuTe object has no initialization entry point")
    status = ctypes.CFUNCTYPE(ctypes.c_int32)(address)()
    if status:
        raise RuntimeError(f"prepared CuTe CUDA initialization failed: {status}")


def unload(module):
    """Release the same once-owned library; a later engine destructor is a no-op."""
    contract()
    address = module.engine.lookup("__cute_internal_func_mlops_cuda_unload")
    if not address:
        raise ValueError("prepared CuTe object has no release entry point")
    ctypes.CFUNCTYPE(None)(address)()
