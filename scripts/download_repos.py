# -*- coding: utf-8 -*-
"""Shallow-clone repos listed in configs/repos.yaml into data/raw/<split>/<name>,
then record the resolved commit SHA of each into configs/repos.lock.yaml so the
exact snapshot used for the dataset is reproducible later.
"""
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "repos.yaml"
LOCK = ROOT / "configs" / "repos.lock.yaml"
RAW_DIR = ROOT / "data" / "raw"


def run(cmd, cwd=None):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{result.stderr}")
    return result.stdout.strip()


def clone_repo(name, url, dest):
    if dest.exists():
        print(f"[skip] {name} already exists at {dest}")
    else:
        print(f"[clone] {name} <- {url}")
        run(["git", "clone", "--depth", "1", url, str(dest)])
    sha = run(["git", "rev-parse", "HEAD"], cwd=dest)
    print(f"[done] {name} @ {sha}")
    return sha


def main():
    with open(CONFIG, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    lock = {}
    for split in ("train", "val", "test"):
        lock[split] = []
        for repo in spec.get(split, []):
            name, url, license_ = repo["name"], repo["url"], repo["license"]
            dest = RAW_DIR / split / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                sha = clone_repo(name, url, dest)
            except RuntimeError as e:
                print(f"[ERROR] {name}: {e}", file=sys.stderr)
                continue
            lock[split].append(
                {"name": name, "url": url, "license": license_, "commit": sha}
            )

    with open(LOCK, "w", encoding="utf-8") as f:
        yaml.safe_dump(lock, f, sort_keys=False)
    print(f"\nLock file written: {LOCK}")


if __name__ == "__main__":
    main()
