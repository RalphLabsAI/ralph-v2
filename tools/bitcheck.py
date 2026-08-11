"""Is an 'unpacked' safetensors model REALLY low-bit, or just the full-precision original?

The whole Bonsai comparison turns on this. `Ternary-Bonsai-8B-unpacked` scoring 88.5% on GSM8K is
a remarkable ternary result — or a meaningless one, if 'unpacked' means the model before it was
compressed. The dtype header cannot answer it: ternary values stored in bf16 look exactly like
bf16, which is precisely the deception `eval/bitrate` exists to catch on miner submissions.

So count DISTINCT VALUES in the actual bytes. A ternary tensor holds three levels (times a
per-group scale); a real bf16 tensor holds thousands. One range request per tensor, no download.
"""
import json
import struct
import sys
import urllib.request

def get(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}",
                                               "User-Agent": "ralph-bitcheck"})
    return urllib.request.urlopen(req, timeout=90).read()


def files(repo):
    u = f"https://huggingface.co/api/models/{repo}/tree/main?recursive=1"
    r = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "ralph"}),
                               timeout=60)
    return json.loads(r.read().decode())


def bf16_vals(b):
    """bf16 is the top 16 bits of an f32."""
    out = []
    for i in range(0, len(b) - 1, 2):
        u = struct.unpack("<H", b[i:i + 2])[0]
        out.append(struct.unpack("<f", struct.pack("<I", u << 16))[0])
    return out


def f16_vals(b):
    return [struct.unpack("<e", b[i:i + 2])[0] for i in range(0, len(b) - 1, 2)]


def check(repo, max_tensors=3):
    print("=" * 74)
    print(repo)
    tree = files(repo)
    sts = sorted([f["path"] for f in tree if f["path"].endswith(".safetensors")])
    if not sts:
        print("  no safetensors")
        return
    url = f"https://huggingface.co/{repo}/resolve/main/{sts[0]}"
    n = struct.unpack("<Q", get(url, 0, 7))[0]
    hdr = json.loads(get(url, 8, 8 + n - 1).decode())
    picked = 0
    for name, meta in hdr.items():
        if name == "__metadata__" or not isinstance(meta, dict):
            continue
        if "weight" not in name or len(meta.get("shape", [])) != 2:
            continue
        dtype = meta["dtype"]
        if dtype not in ("BF16", "F16"):
            print(f"  {name[:44]:<44} dtype={dtype}  (not 16-bit, skipping)")
            continue
        s, e = meta["data_offsets"]
        base = 8 + n
        take = min(8192, e - s)                       # 4096 values is plenty to characterise
        raw = get(url, base + s, base + s + take - 1)
        vals = bf16_vals(raw) if dtype == "BF16" else f16_vals(raw)
        uniq = sorted(set(round(v, 6) for v in vals))
        print(f"  {name[:44]:<44} dtype={dtype} shape={meta['shape']}")
        print(f"      {len(vals)} values sampled -> {len(uniq)} distinct overall")

        # THE DECISIVE TEST IS PER GROUP, NOT OVERALL. Ternary is three LEVELS times a per-group
        # scale, so a sample spanning many groups shows 3 x n_groups distinct values and looks
        # like a 6-bit codebook. Inside one group it must collapse to ~3 (often 2, when no weight
        # in that group took one of the levels). Counting across the whole sample cannot tell
        # ternary from 6-bit; counting inside a window can.
        for win in (16, 32, 64, 128):
            counts = []
            for i in range(0, min(len(vals), 2048), win):
                chunk = vals[i:i + win]
                if len(chunk) == win:
                    counts.append(len(set(round(v, 6) for v in chunk)))
            if counts:
                counts.sort()
                med = counts[len(counts) // 2]
                print(f"      window={win:<4} distinct/window: min={counts[0]} "
                      f"median={med} max={counts[-1]}")
        # a ternary group collapses to <=3 levels (plus sign) inside its own window
        w64 = [len(set(round(v, 6) for v in vals[i:i + 64]))
               for i in range(0, min(len(vals), 2048), 64) if len(vals[i:i + 64]) == 64]
        med64 = sorted(w64)[len(w64) // 2] if w64 else 0
        if med64 <= 4:
            verdict = "TERNARY-like: ~3 levels inside each group"
        elif med64 <= 20:
            verdict = f"LOW-BIT: ~{med64} levels per 64-weight group (~{max(1, med64).bit_length()} bits)"
        else:
            verdict = f"NOT low-bit: {med64} distinct values inside a single 64-weight group"
        print(f"      -> {verdict}")
        picked += 1
        if picked >= max_tensors:
            break


if __name__ == "__main__":
    for repo in sys.argv[1:]:
        try:
            check(repo)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
