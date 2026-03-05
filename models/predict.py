"""
Quick-start script for flood prediction.

Usage:
    # Single file
    python predict.py path/to/site.geojson

    # Folder of files
    python predict.py path/to/folder/

    # Multiple files
    python predict.py site1.geojson site2.geojson site3.geojson
"""

import sys
import json
from pathlib import Path

from flood_model.config import Config
from flood_model.inference import load_model_and_normalizer, predict_single


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <file.geojson | folder/> [...]")
        sys.exit(1)

    # Load ensemble once
    cfg = Config()
    models, normalizer, threshold = load_model_and_normalizer(cfg)

    # Collect all .geojson files from arguments
    files = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_dir():
            files.extend(sorted(p.glob("*.geojson")))
        elif p.is_file():
            files.append(p)
        else:
            print(f"⚠  Skipping {arg} (not found)")

    if not files:
        print("No GeoJSON files found.")
        sys.exit(1)

    # Predict
    results = []
    for f in files:
        result = predict_single(f, cfg, models, normalizer, threshold)
        results.append(result)
        icon = "🌊" if result["prediction"] == 1 else "✅"
        print(f"  {icon}  {result['probability']:.4f}  {result['file']}")

    # Summary
    n_flood = sum(r["prediction"] for r in results)
    print(f"\n{'═' * 40}")
    print(f"  Total: {len(results)}  |  Flood: {n_flood}  |  Safe: {len(results) - n_flood}")

    # Save results
    out = Path("predictions.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {out}")


if __name__ == "__main__":
    main()
