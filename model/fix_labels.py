# fix_labels.py
import glob, re

# adjust this glob if your labels live elsewhere
for fn in glob.glob("data/**/labels/*.txt", recursive=True):
    with open(fn, "r", encoding="utf8", errors="ignore") as f:
        raw = f.read().replace("\x00", "")
    good = []
    for line in raw.splitlines():
        parts = line.strip().split()
        # must be exactly 5 numeric tokens
        if len(parts) == 5 and all(re.fullmatch(r"-?\d+(\.\d+)?", p) for p in parts):
            good.append(" ".join(parts))
    # overwrite (if no good lines, file becomes empty = “no objects”)
    with open(fn, "w") as f:
        f.write("\n".join(good))
    print(f"Fixed {fn}: {len(good)} valid lines kept")
