"""Export an offline interactive teaching dashboard from saved RL checkpoints."""
import argparse
import json
from pathlib import Path
import sys
import webbrowser

ROOT = Path(__file__).resolve().parent
if (ROOT / ".deps").exists():
    sys.path.insert(0, str(ROOT / ".deps"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123, help="Environment randomness for replayed episodes")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu",
                        help="Model inspection device; browser playback never needs CUDA")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "dashboard.html")
    parser.add_argument("--open", action="store_true", help="Open the generated dashboard in the default browser")
    parser.add_argument("--skip-world-model", action="store_true")
    args = parser.parse_args()
    from warehouse.inspection import build_dashboard_data
    from warehouse.device import resolve_device
    try:
        device = resolve_device(args.device)
        data = build_dashboard_data(ROOT, seed=args.seed, device=device,
                                    include_world=not args.skip_world_model)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    assets = ROOT / "warehouse" / "visualizer"
    # Inline data/assets: file:// playback needs no server, CDN, or network.
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    html = (assets / "dashboard.html").read_text(encoding="utf-8")
    html = html.replace("/* INLINE_STYLES */", (assets / "style.css").read_text(encoding="utf-8"))
    html = html.replace("/* INLINE_DATA */", "window.RL_DATA = " + payload + ";")
    html = html.replace("/* INLINE_APP */", (assets / "app.js").read_text(encoding="utf-8"))
    html = html.replace("/* INLINE_TEXTBOOK */", (assets / "textbook.js").read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Exported {len(data['runs'])} policies; episode seed={args.seed}; inference={device}")
    print(args.output.resolve())
    for warning in data["warnings"]:
        print("Warning:", warning)
    if args.open:
        webbrowser.open(args.output.resolve().as_uri())


if __name__ == "__main__":
    main()
