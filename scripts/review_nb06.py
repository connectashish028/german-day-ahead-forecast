"""Extract printable outputs from notebooks/06_weather_impact.ipynb so we can
review the actual numbers without rendering plotly."""
import json
from pathlib import Path

nb = json.loads(Path("notebooks/06_weather_impact.ipynb").read_text(encoding="utf-8"))
for i, cell in enumerate(nb["cells"], start=1):
    if cell["cell_type"] != "code":
        continue
    print(f"\n=== Cell {i}  ({cell['execution_count']}) ===")
    src_preview = "".join(cell["source"])[:120].replace("\n", " ¶ ")
    print(f"src> {src_preview}…")
    for out in cell.get("outputs", []):
        kind = out.get("output_type")
        if kind == "stream":
            print(f"[stdout]")
            print("  " + "  ".join(out["text"]) if isinstance(out["text"], list) else "  " + out["text"])
        elif kind == "execute_result":
            data = out.get("data", {})
            if "text/plain" in data:
                txt = data["text/plain"]
                if isinstance(txt, list):
                    txt = "".join(txt)
                print(f"[result/text]")
                print("  " + txt.replace("\n", "\n  "))
        elif kind == "error":
            print(f"[error] {out.get('ename')}: {out.get('evalue')}")
