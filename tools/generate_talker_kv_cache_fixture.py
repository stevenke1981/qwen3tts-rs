import argparse, json, hashlib, struct
from pathlib import Path

EXPECTED = {
    "snapshot_revision": "5d83992436eae1d760afd27aff78a71d676296fc",
    "official_qwen_revision": "022e286b98fbec7e1e916cb940cdf532cd9f488e",
    "qwentts_cpp_revision": "82cd05b9f3a175612dc89fd6943e610fab096ef5",
}
ORACLE = {
    "next-emb-step0.bin": "E12A11499F4007FC80FDEAB1E891BC8532EE69906F2B35C9701D7C0DD7F5422C",
    "talker-hidden-step1.bin": "E092721E0B377EC02D1A08306E8A474140B3BF199E87DA1060FF229297315EAD",
    "codes-step0.bin": "C0C5B11883C53C6DF63E97BCB2E422F115EE5CD4D1FB6EC403397632A5668D27",
}

def tensor_header(path):
    raw = path.read_bytes()
    rank = struct.unpack_from("<I", raw, 0)[0]
    dims = struct.unpack_from("<" + "I" * rank, raw, 4)
    return rank, list(dims), raw

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--check", type=Path, required=True); ap.add_argument("--metrics", type=Path); a=ap.parse_args()
    d=json.loads(a.check.read_text(encoding="utf-8"))
    assert d.get("version")==1 and d.get("fixture_id")=="p03-talker-kv-cache-real"
    for k,v in EXPECTED.items(): assert d.get(k)==v, (k,d.get(k),v)
    assert d.get("min_cosine")==0.999 and d.get("max_abs_limit")==0.001
    oracle = Path("artifacts/alignment/P03/P03-T02/qwentts-direct-oracle/raw_tensor_dump")
    for name, expected in ORACLE.items():
        p = oracle / name
        assert p.is_file(), f"FIXTURE_MISSING: {p}"
        digest = hashlib.sha256(p.read_bytes()).hexdigest().upper()
        assert digest == expected, (name, digest, expected)
    rank, dims, raw = tensor_header(oracle / "next-emb-step0.bin")
    assert (rank, dims) == (1, [1024])
    rank, dims, _ = tensor_header(oracle / "talker-hidden-step1.bin")
    assert (rank, dims) == (1, [1024])
    rank, dims, raw = tensor_header(oracle / "codes-step0.bin")
    assert (rank, dims) == (1, [16])
    codes = struct.unpack_from("<" + "f" * 16, raw, 8)
    assert codes[0] == 1995, f"codes-step0 semantic token mismatch: {codes[0]}"
    if a.metrics:
        m=json.loads(a.metrics.read_text(encoding="utf-8"))
        assert m.get("schema_version")==1 and len(m.get("layers", []))==28
        assert m.get("sequence")==11 and "provenance" in m
        for key in ("next_embedding", "qwentts_hidden_step1", "final_hidden", "logits"):
            assert m[key]["cosine"] >= 0.999 and m[key]["max_abs"] <= 0.001
        for layer in m["layers"]:
            assert layer["k_cosine"] >= 0.999 and layer["k_max_abs"] <= 0.001
            assert layer["v_cosine"] >= 0.999 and layer["v_max_abs"] <= 0.001
    print("p03_talker_kv_cache_fixture: PASS")
if __name__ == "__main__": main()
