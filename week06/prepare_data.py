"""Workspace path and read-only integrity inventory of reference weeks."""
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def inventory():
    return {str(p.relative_to(ROOT.parent)):[p.stat().st_size,p.stat().st_mtime_ns]
            for folder in ROOT.parent.glob('week*') if folder.is_dir() and folder!=ROOT
            for p in folder.rglob('*') if p.is_file()}
