"""输出文件命名：``<原名>_<n>.<后缀>``，n 从 0 开始取第一个未占用的编号。

源文件名结尾已有的 ``_<数字>`` 会先去掉，避免把成片再次换脸时叠加成
``img_1_0.png`` / ``img_1_0_1.png`` 这类重复后缀。
"""

from __future__ import annotations

import re
from pathlib import Path

_NUMERIC_SUFFIX = re.compile(r"(?:_\d+)+$")


def base_stem(source: Path | str) -> str:
    """源文件名去掉结尾的 ``_<数字>``；整名都是编号时保留原名。"""
    stem = Path(source).stem
    stripped = _NUMERIC_SUFFIX.sub("", stem)
    return stripped or stem


def next_output_path(
    source: Path | str,
    suffix: str = ".png",
    directory: Path | str | None = None,
) -> Path:
    """返回 ``<目录>/<去编号原名>_<n><后缀>`` 中第一个不存在的路径。

    - n 从 0 开始：``img.jpg`` → ``img_0.png``；已存在则 ``img_1.png``；
    - 源文件名带编号时先归一：``img_21.jpg`` → ``img_0.png`` / ``img_1.png``…
    - 跳过与源文件同名的候选，避免覆盖输入文件。
    """
    source = Path(source)
    folder = Path(directory) if directory is not None else source.parent
    if suffix and not suffix.startswith("."):
        suffix = "." + suffix
    stem = base_stem(source)
    source_resolved = source.resolve()
    n = 0
    while True:
        candidate = folder / f"{stem}_{n}{suffix}"
        if not candidate.exists() and candidate.resolve() != source_resolved:
            return candidate
        n += 1
