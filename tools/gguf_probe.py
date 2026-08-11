"""Read a GGUF's metadata from the first few MB over HTTP. No download, no credentials.

Every submission advertises, in plaintext, the quantisation type it used AND the imatrix
calibration dataset it was fitted on. That is a free read on a public artifact, and it is the
single most informative thing about how a crowned model was built.
"""
import json
import struct
import sys
import urllib.request

# GGUF value types
(U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64) = range(13)

FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
    16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
    22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
    28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
    36: "TQ1_0", 37: "TQ2_0",
}


def fetch(url, nbytes):
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{nbytes - 1}",
                                               "User-Agent": "ralph-gguf-probe"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


class R:
    def __init__(self, b):
        self.b, self.i = b, 0

    def take(self, n):
        if self.i + n > len(self.b):
            raise EOFError("ran past the fetched window")
        v = self.b[self.i:self.i + n]
        self.i += n
        return v

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        return self.take(self.u64()).decode("utf-8", "replace")

    def value(self, t):
        if t == STRING:
            return self.string()
        if t in (U32, I32):
            return struct.unpack("<i" if t == I32 else "<I", self.take(4))[0]
        if t in (U64, I64):
            return struct.unpack("<q" if t == I64 else "<Q", self.take(8))[0]
        if t in (U8, I8, BOOL):
            return struct.unpack("<b" if t == I8 else "<B", self.take(1))[0]
        if t in (U16, I16):
            return struct.unpack("<h" if t == I16 else "<H", self.take(2))[0]
        if t == F32:
            return struct.unpack("<f", self.take(4))[0]
        if t == F64:
            return struct.unpack("<d", self.take(8))[0]
        if t == ARRAY:
            et, n = self.u32(), self.u64()
            if et == STRING:                       # token lists are huge and uninteresting
                if n > 64:
                    for _ in range(n):
                        self.take(self.u64())
                    return f"<{n} strings>"
                return [self.string() for _ in range(n)]
            if n <= 16:
                return [self.value(et) for _ in range(n)]
            # SKIP THE BYTES, do not merely decline to return them. Returning a placeholder without
            # advancing left the reader mid-array, so every field after it decoded as garbage —
            # a 151k-element token_type array turned the rest of the header into noise and the
            # parse "ran out of window" no matter how much window it was given.
            width = {U8: 1, I8: 1, BOOL: 1, U16: 2, I16: 2, U32: 4, I32: 4, F32: 4,
                     U64: 8, I64: 8, F64: 8}.get(et)
            if width is None:
                raise ValueError(f"cannot skip array of type {et}")
            self.take(n * width)
            return f"<{n} values>"
        raise ValueError(f"unknown gguf type {t}")


def read_header(url, window=8 << 20):
    """Grow the window until the metadata block fits. Qwen3 ships a ~151k-token vocabulary plus
    merges, so the KV section alone runs to tens of MB — but it is still a rounding error against
    downloading a 4.6 GB model to ask which quantiser made it."""
    last = None
    for w in (window, 4 * window, 16 * window, 64 * window):
        try:
            return _parse(fetch(url, w))
        except EOFError as e:
            last = e
    raise last


def _parse(blob):
    r = R(blob)
    if r.take(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    ver, n_tensors, n_kv = r.u32(), r.u64(), r.u64()
    kv = {}
    for _ in range(n_kv):
        k = r.string()
        kv[k] = r.value(r.u32())
    return {"version": ver, "n_tensors": n_tensors, "kv": kv}


def files_of(repo, rev="main"):
    url = f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=1"
    with urllib.request.urlopen(urllib.request.Request(
            url, headers={"User-Agent": "ralph-gguf-probe"}), timeout=60) as r:
        return json.loads(r.read().decode())


def report(label, repo, rev="main"):
    print("=" * 78)
    print(f"{label}\n  hf://{repo}@{rev}")
    try:
        tree = files_of(repo, rev)
    except Exception as e:
        print(f"  could not list files: {type(e).__name__}: {e}")
        return
    ggufs = [f for f in tree if f.get("path", "").lower().endswith(".gguf")]
    for f in tree:
        print(f"    {f.get('path'):<48} {(f.get('size') or 0) / 1e9:6.2f} GB")
    if not ggufs:
        print("  no gguf")
        return
    g = ggufs[0]
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{g['path']}"
    try:
        h = read_header(url)
    except Exception as e:
        print(f"  header unreadable: {type(e).__name__}: {e}")
        return
    kv = h["kv"]
    ft = kv.get("general.file_type")
    print(f"\n  tensors           : {h['n_tensors']}")
    print(f"  file_type         : {ft} ({FILE_TYPES.get(ft, '?')})")
    for key in ("general.architecture", "general.name", "general.basename",
                "general.size_label", "general.quantized_by", "general.finetune",
                "quantize.imatrix.file", "quantize.imatrix.dataset",
                "quantize.imatrix.entries_count", "quantize.imatrix.chunks_count"):
        if key in kv:
            print(f"  {key:<18}: {kv[key]}")
    ctx = [k for k in kv if k.endswith(("context_length", "block_count",
                                        "embedding_length", "head_count"))]
    for k in sorted(ctx):
        print(f"  {k:<18}: {kv[k]}")


if __name__ == "__main__":
    for line in sys.argv[1:]:
        label, repo_rev = line.split("=", 1)
        repo, _, rev = repo_rev.partition("@")
        report(label, repo, rev or "main")
