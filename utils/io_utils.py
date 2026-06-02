"""IO 工具函数。"""
from pathlib import Path


def get_next_test_dir(base_dir: Path) -> Path:
    """在 base_dir 下查找已有数字编号文件夹，返回下一个递增编号路径。"""
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = [d for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    next_id = max([int(d.name) for d in existing], default=0) + 1
    return base_dir / str(next_id)
