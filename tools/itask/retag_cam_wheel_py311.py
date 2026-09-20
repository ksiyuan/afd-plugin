#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Retag the CAM Python binding wheel from CPython 3.12 to CPython 3.11.

Local workaround, not an upstream artifact. The CAM operator package ships
``umdk_cam_op_lib`` as a single compiled extension built for CPython 3.12
(``umdk_cam_op_lib.cpython-312-aarch64-linux-gnu.so``). That module does not
link ``libpython``; its CPython footprint is 118 symbols that all exist in
3.11 as well (``PyFrame_GetBack``, ``PyCMethod_New``, ``PyThread_tss_*``,
``_PyObject_GetDictPtr``, ...). Everything else it resolves comes from their
torch stack (``libtorch.so``, ``libtorch_npu.so``, ``libascendcl.so``), so
the wheel tag is the only Python-version coupling.

This rewrites the wheel tag and the extension file name, and regenerates
``RECORD``, so ``pip install`` accepts it on Python 3.11. Verify the import
afterwards: the four ops must be registered under ``torch.ops.umdk_cam_op_lib``.

Usage:
    python3 tools/itask/retag_cam_wheel_py311.py --output-dir /tmp/cam311
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import zipfile
from pathlib import Path

DEFAULT_WHEEL = Path(
    "afd_plugin/connectors/npu/bin/"
    "umdk_cam_op_lib-209.0.0b1-cp312-cp312-linux_aarch64.whl",
)
NEW_ABI = "cp311-cp311"
NEW_TAG = f"{NEW_ABI}-linux_aarch64"


def record_line(name: str, data: bytes | None) -> str:
    if data is None:
        return f"{name},,"
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"{name},sha256={digest.decode()},{len(data)}"


def retag(source: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / source.name.replace("cp312-cp312", NEW_ABI)
    entries: list[tuple[str, bytes]] = []

    with zipfile.ZipFile(source) as archive:
        for item in archive.infolist():
            name = item.filename.replace("cpython-312", "cpython-311")
            data = archive.read(item.filename)
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
