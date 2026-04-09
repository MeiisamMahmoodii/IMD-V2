import hashlib
import tarfile
from datetime import UTC, datetime
from pathlib import Path

root = Path("experiments/results")
archive_dir = root / "data_archives"
archive_dir.mkdir(parents=True, exist_ok=True)
ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
archive = archive_dir / f"final_results_{ts}.tar.gz"
manifest = archive_dir / f"final_results_{ts}_manifest.sha256"

with tarfile.open(archive, "w:gz") as tar:
    for p in sorted(root.rglob("*")):
        if p.is_file() and "data_archives" not in str(p):
            tar.add(p, arcname=p.relative_to(root))

lines = []
for p in sorted(root.rglob("*")):
    if p.is_file() and "data_archives" not in str(p):
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{h}  {p.relative_to(root)}")
manifest.write_text("\n".join(lines), encoding="utf-8")

appendix = root / "RESULTS_APPENDIX.md"
jsons = list(root.glob("exp_*.json"))
csvs = list((root / "reports").glob("*.csv"))
appendix.write_text(
    "# Results Appendix\n\n"
    f"- Experiment JSON count: {len(jsons)}\n"
    f"- Report CSV count: {len(csvs)}\n"
    f"- Archive: {archive.name}\n"
    f"- Manifest: {manifest.name}\n",
    encoding="utf-8",
)
print(str(archive))

