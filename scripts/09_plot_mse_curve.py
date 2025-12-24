from pathlib import Path
import re
import matplotlib.pyplot as plt

def main():
    project_root = Path(__file__).resolve().parents[1]
    report_path = project_root / "results" / "reports" / "rom_eval.txt"

    text = report_path.read_text(encoding="utf-8")

    # Extract "step i: value" lines
    pattern = r"step\s+(\d+):\s+([0-9.]+)"
    matches = re.findall(pattern, text)

    steps = [int(s) for s, _ in matches]
    mse = [float(v) for _, v in matches]

    out_png = project_root / "results" / "figures" / "mse_vs_step.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(steps, mse, marker="o")
    plt.xlabel("Prediction step ahead")
    plt.ylabel("Mean MSE (velocity_norm)")
    plt.title("ROM error growth during rollout")
    plt.grid(True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")

    print("✅ Saved:", out_png)

if __name__ == "__main__":
    main()
