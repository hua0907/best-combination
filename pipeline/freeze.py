"""Freeze the roundtable engine into this project.

    python pipeline/freeze.py            # once
    python pipeline/freeze.py --force    # replace the frozen copy: a new strategy version, re-validate before trading

Copies from config.json's engine_source (D:/skills/stock/stock-roundtable) the engine package, run_backtest.py,
fetch_xdxr.py, run_day.py, optimal_combo.py and the docs, unchanged, plus the nine debaters' SKILL.md from the folder
next to it (the engine's backtest audit fingerprints them). skills/FROZEN.json records a SHA-256 for every frozen file;
every run compares against it and the page warns when something changed.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

from common import CONFIG, ENGINE_DIR, SKILLS, now, sha256

FILES = ['run_backtest.py', 'fetch_xdxr.py', 'run_day.py', 'optimal_combo.py', 'SKILL.md', 'README.md', '最优组合策略.md']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    src = Path(CONFIG['engine_source'])
    if (SKILLS / 'FROZEN.json').exists() and not args.force:
        sys.exit('已经冻结过；要换成新版本请加 --force（换版本后必须重新验证）')
    (ENGINE_DIR / 'engine').mkdir(parents=True, exist_ok=True)
    frozen = []
    for p in sorted((src / 'engine').glob('*.py')):
        shutil.copy2(p, ENGINE_DIR / 'engine' / p.name)
        frozen.append(f'stock-roundtable/engine/{p.name}')
    for name in FILES:
        if (src / name).exists():
            shutil.copy2(src / name, ENGINE_DIR / name)
            frozen.append(f'stock-roundtable/{name}')
    sys.path.insert(0, str(ENGINE_DIR))
    from engine import config as ecfg
    from engine.debaters import all_debaters
    for d in all_debaters():
        (SKILLS / d.key).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src.parent / d.key / 'SKILL.md', SKILLS / d.key / 'SKILL.md')
        frozen.append(f'{d.key}/SKILL.md')
    manifest = {'frozen_at': now().isoformat(timespec='seconds'), 'source': str(src), 'version': ecfg.VERSION,
                'data_version': ecfg.DATA_VERSION,
                'note': '最优组合（2026-10-17 用户指定）用的引擎和规则；冻结后不改。换版本要用 --force 重新冻结并重新验证。',
                'files': {name: sha256(SKILLS / name) for name in frozen}}
    (SKILLS / 'FROZEN.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'frozen {len(frozen)} files of {ecfg.VERSION} into {SKILLS}')


if __name__ == '__main__':
    main()
