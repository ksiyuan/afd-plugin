#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Retag the CAM Python binding wheel from CPython 3.12 to CPython 3.11.

Local workaround, not an upstream artifact. The CAM operator package ships
``umdk_cam_op_lib`` as a single compiled extension built for CPython 3.12
(``umdk_cam_op_lib.cpython-312-aarch64-linux-gnu.so``). Three things pin it to
3.12, and none of them is a C-ABI problem: the module does not link
``libpython``, the ``PyModuleDef`` layout it uses matches 3.11's (``m_name``
at +40, ``m_slots`` at +72), and every CPython symbol it imports exists in
3.11. Everything else it resolves comes from the torch stack
(``libtorch.so``, ``libtorch_npu.so``, ``libascendcl.so``).

1. The wheel tag and the extension file name, for ``pip`` and the import
   system.
2. The runtime version guard in ``comm_operator/pybind/pybind.cpp``, which
   refuses to load unless ``Py_GetVersion()`` starts with the version the
   module was built with. The same rodata literal feeds the error message, so
   rewriting it keeps the diagnostic truthful.
3. ``Py_mod_multiple_interpreters`` (slot ID 3, added in CPython 3.12), which
   3.11's import machinery rejects with

       SystemError: module umdk_cam_op_lib uses unknown slot ID 3

   The slot table lives in .bss and is filled at run time, so the fix is the
   store that puts ``3`` into it: the ``mov w4, #3`` immediately before that
   store is rewritten to ``mov w4, #0``, which turns the entry into the
   ``{0, NULL}`` terminator.

Both in-module edits are located by content rather than by hard-coded offset,
and fail loudly when the module does not match. Each changes a single byte.
After installing, ``import torch, torch_npu, umdk_cam_op_lib`` must expose the
four ops under ``torch.ops.umdk_cam_op_lib``, with no ``LD_PRELOAD`` needed.

Usage:
    python3 tools/itask/retag_cam_wheel_py311.py --output-dir /tmp/cam311
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import struct
import zipfile
from pathlib import Path

DEFAULT_WHEEL = Path(
    "afd_plugin/connectors/npu/bin/"
    "umdk_cam_op_lib-209.0.0b1-cp312-cp312-linux_aarch64.whl",
)
NEW_ABI = "cp311-cp311"
NEW_TAG = f"{NEW_ABI}-linux_aarch64"

GUARD_MESSAGE = b"Python version mismatch: module was compiled for Python"
GUARD_SEARCH_WINDOW = 64

# mov w5, #2 / mov w4, #3 / mov w6, #1 / mov x0, x19: the slot-table setup,
# where the second instruction puts Py_mod_multiple_interpreters (3) into the
# slot ID that is stored right after.
SLOT_ANCHOR = bytes.fromhex("450080526400805226008052e00313aa")
SLOT_MOV_OFFSET = 4
SLOT_MOV_W4_3 = 0x52800064
SLOT_MOV_W4_0 = 0x52800004


def record_line(name: str, data: bytes | None) -> str:
    if data is None:
        return f"{name},,"
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"{name},sha256={digest.decode()},{len(data)}"


def clear_python_guard(data: bytes) -> bytes:
    """Point the extension's built-in version guard at CPython 3.11."""
    message = data.find(GUARD_MESSAGE)
    if message < 0:
        raise RuntimeError("version-guard message not found in the module")
    window_start = max(0, message - GUARD_SEARCH_WINDOW)
    literal = data.rfind(b"3.12\0", window_start, message)
    if literal < 0:
        raise RuntimeError("no '3.12' literal next to the version guard")
    print(f"  guard literal at file offset {literal}: 3.12 -> 3.11")
    return data[:literal] + b"3.11" + data[literal + 4 :]


def clear_multiple_interpreters_slot(data: bytes) -> bytes:
    """Turn the Py_mod_multiple_interpreters slot into the terminator."""
    hits: list[int] = []
    start = data.find(SLOT_ANCHOR)
    while start != -1:
        hits.append(start)
        start = data.find(SLOT_ANCHOR, start + 1)
    if len(hits) != 1:
        raise RuntimeError(f"expected one slot-table anchor, found {len(hits)}")
    offset = hits[0] + SLOT_MOV_OFFSET
    instruction = struct.unpack_from("<I", data, offset)[0]
    if instruction != SLOT_MOV_W4_3:
        raise RuntimeError(
            f"unexpected slot instruction {instruction:#010x} at {offset}",
        )
    print(
        f"  Py_mod_multiple_interpreters write at file offset {offset}: "
        f"{SLOT_MOV_W4_3:#010x} -> {SLOT_MOV_W4_0:#010x}",
    )
    return data[:offset] + struct.pack("<I", SLOT_MOV_W4_0) + data[offset + 4 :]


def retag(source: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / source.name.replace("cp312-cp312", NEW_ABI)
    entries: list[tuple[str, bytes]] = []

    with zipfile.ZipFile(source) as archive:
        for item in archive.infolist():
            name = item.filename.replace("cpython-312", "cpython-311")
            data = archive.read(item.filename)
            if name.endswith(".so"):
                data = clear_python_guard(data)
                data = clear_multiple_interpreters_slot(data)
            if name.endswith(".dist-info/WHEEL"):
                data = data.replace(b"cp312-cp312", NEW_TAG.encode())
            if name.endswith(".dist-info/RECORD"):
                continue
            entries.append((name, data))

    dist_info = next(
        name for name, _ in entries if name.endswith(".dist-info/top_level.txt")
    ).rsplit("/", 1)[0]
    record_name = f"{dist_info}/RECORD"
    record = [record_line(name, data) for name, data in entries]
    record.append(record_line(record_name, None))
    entries.append((record_name, ("\n".join(record) + "\n").encode()))

    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries:
            archive.writestr(name, data)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, default=DEFAULT_WHEEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    target = retag(args.wheel, args.output_dir)
    print(f"retagged wheel: {target}")
    with zipfile.ZipFile(target) as archive:
        for info in archive.infolist():
            print(f"  {info.file_size:>9}  {info.filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
