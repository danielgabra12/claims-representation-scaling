"""Render sql/01_extract.sql -> sql/01_extract.rendered.sql from schema_config.yaml.

Flattens the config into ${...} substitution keys. Run after Gate A:
    python sql/render.py
Fails loudly if any placeholder is left unresolved or any value is still a
PLACEHOLDER, so you can't accidentally paste an un-filled query into Snowflake.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from string import Template

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "schema_config.yaml"
# Templates to render (input -> output). 02 is the combined ML+epi extraction.
TEMPLATES = {
    ROOT / "sql" / "01_extract.sql": ROOT / "sql" / "01_extract.rendered.sql",
    ROOT / "sql" / "02_extract_ml.sql": ROOT / "sql" / "02_extract_ml.rendered.sql",
}


def flatten(cfg: dict) -> dict[str, str]:
    """Build the flat key->value map the templates expect."""
    mapping: dict[str, str] = {}
    mapping["database"] = cfg["database"]
    mapping["schema"] = cfg["schema"]
    mapping.update(cfg["tables"])      # enrollment, claims, pharmacy, demographics
    mapping.update(cfg["columns"])     # patient_id, claim_date, icd_code, ...
    mapping.update({k: str(v) for k, v in cfg["study"].items()})
    return {k: str(v) for k, v in mapping.items()}


def render_one(src: Path, out: Path, mapping: dict[str, str]) -> int:
    template = Template(src.read_text())
    try:
        rendered = template.substitute(mapping)
    except KeyError as e:
        print(f"[{src.name}] references ${{{e.args[0]}}} with no config value.",
              file=sys.stderr)
        return 1
    leftovers = re.findall(r"\$\{[^}]+\}", rendered)
    if leftovers:
        print(f"[{src.name}] unresolved placeholders: {sorted(set(leftovers))}",
              file=sys.stderr)
        return 1
    out.write_text(rendered)
    print(f"Rendered {src.name} -> {out.relative_to(ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Render extraction SQL from config.")
    ap.add_argument("--only", help="render just this template filename "
                    "(e.g. 02_extract_ml.sql)")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text())
    mapping = flatten(cfg)

    if any("PLACEHOLDER" in v for v in mapping.values()):
        print("REFUSING TO RENDER: schema_config.yaml still has PLACEHOLDER "
              "values. Fill/confirm the real IQVIA names first.", file=sys.stderr)
        return 1

    rc = 0
    for src, out in TEMPLATES.items():
        if args.only and src.name != args.only:
            continue
        if not src.exists():
            continue
        rc |= render_one(src, out, mapping)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
