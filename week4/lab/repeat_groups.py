"""같은 설정 반복 run들의 fp32 EMA 해시·첫 run 대비 최대 절대차·마지막 지표. 사용: repeat_groups.py <runs> <out.json> <run>..."""
import sys, hashlib, torch, json
from pathlib import Path
runs = Path(sys.argv[1]); out = {}
def th(sd):
    h = hashlib.sha256()
    for k in sorted(sd): h.update(sd[k].numpy().tobytes())
    return h.hexdigest()[:12]
names = sys.argv[3:]
base = torch.load(runs/names[0]/"fp32/ema.pt")
for n in names:
    sd = torch.load(runs/n/"fp32/ema.pt")
    d = max((sd[k]-base[k]).abs().max().item() for k in sd if sd[k].is_floating_point())
    m = open(runs/n/"results.csv").read().strip().splitlines()[-1].split(",")
    out[n] = {"fp32_ema_sha12": th(sd), "max_abs_diff_vs_first": d, "last_mAP50-95": m[8], "last_box_loss": m[2]}
    print(n, out[n])
json.dump(out, open(sys.argv[2], "w"), indent=1)
