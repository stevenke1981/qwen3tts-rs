#!/usr/bin/env python3
"""Independent, offline qwentts.cpp sampling corpus generator."""
import argparse, hashlib, json, math, struct, copy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "fixtures/alignment/p02_deterministic_token_sequences.json"
REV = "82cd05b9f3a175612dc89fd6943e610fab096ef5"
QWEN_REV = "022e286b98fbec7e1e916cb940cdf532cd9f488e"

def f32(x): return struct.unpack("<f", struct.pack("<f", float(x)))[0]
def f32_sum(values):
    total = 0.0
    for value in values: total = f32(total + value)
    return total
def philox(seed, subseq):
    M0, M1, W0, W1 = 0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85
    lo, hi = seed & 0xffffffff, (seed >> 32) & 0xffffffff
    s = [0, 0, subseq & 0xffffffff, (subseq >> 32) & 0xffffffff]
    def rnd(a, k0, k1):
        p0, p1 = M0*a[0], M1*a[2]
        return [((p1>>32)^a[1]^k0)&0xffffffff, p1&0xffffffff,
                ((p0>>32)^a[3]^k1)&0xffffffff, p0&0xffffffff]
    s = rnd(s, lo, hi)
    k0, k1 = lo, hi
    for _ in range(9):
        k0, k1 = (k0+W0)&0xffffffff, (k1+W1)&0xffffffff
        s = rnd(s, k0, k1)
    return (f32((s[0] + 0.5) * 2.3283064365386963e-10))

def oracle(logits, opts, history, suppress_from=None, allow=None, seed=0, subseq=0):
    vals = [f32(x) for x in logits]
    allowed = lambda i: math.isfinite(vals[i]) and (suppress_from is None or i < suppress_from or i == allow)
    if opts["repetition_penalty"] != 1.0:
        p = f32(opts["repetition_penalty"])
        for i in dict.fromkeys(history):
            if 0 <= i < len(vals): vals[i] = f32(vals[i] * p if vals[i] < 0 else vals[i] / p)
    for i in range(len(vals)):
        if not allowed(i): vals[i] = -math.inf
    cand = [(i, vals[i]) for i in range(len(vals)) if math.isfinite(vals[i])]
    if not cand: raise ValueError("suppression left no candidate")
    cand.sort(key=lambda x:(-x[1],x[0]))
    if not opts["do_sample"]: return cand[0][0], False
    t = f32(opts["temperature"])
    vals = [f32(v/t) if math.isfinite(v) else -math.inf for v in vals]
    cand = [(i, vals[i]) for i in range(len(vals)) if math.isfinite(vals[i])]
    cand.sort(key=lambda x:(-x[1],x[0]))
    if opts["top_k"] and len(cand)>opts["top_k"]:
        # qwentts uses nth_element then masks values strictly below kth score;
        # ties at the boundary remain eligible.
        threshold=sorted((v for _,v in cand), reverse=True)[opts["top_k"]-1]
        vals = [v if v >= threshold else -math.inf for v in vals]
        cand=[item for item in cand if item[1] >= threshold]
    if opts["top_p"] < 1.0:
        m=max(v for v in vals if math.isfinite(v))
        total=f32_sum([f32(math.exp(v-m)) if math.isfinite(v) else 0.0 for v in vals])
        cutoff=m-16.0
        probs=[]
        for i,v in enumerate(vals):
            if math.isfinite(v) and v >= cutoff: probs.append((i,f32(math.exp(v-m)/total)))
            else: vals[i] = -math.inf
        probs.sort(key=lambda x:(-x[1],x[0]))
        cum=0.0
        for pos,(i,p) in enumerate(probs):
            if pos > 0 and cum >= opts["top_p"]: vals[i] = -math.inf
            cum=f32(cum+p)
    m=max(v for v in vals if math.isfinite(v))
    probs=[f32(math.exp(v-m)) if math.isfinite(v) else 0.0 for v in vals]
    total=f32_sum(probs); r=philox(seed,subseq); threshold=f32(r*total); cum=0.0
    for i,p in enumerate(probs):
        cum=f32(cum+p)
        if cum >= threshold: return i, True
    return len(vals)-1, True

def make_case(cid, talker, cp, seed, options, vocab=32):
    calls=[]; frames=[]; history=[]; subseq=0
    for frame in range(2):
        row=[]
        for group in range(16):
            logits=[-10.0]*vocab
            base=(3+frame*5+group*2) % (vocab-2)
            logits[base]=4.0; logits[(base+1)%vocab]=3.0; logits[(base+2)%vocab]=2.0
            sampled = talker if group==0 else cp
            o=dict(options); o["do_sample"]=sampled
            token, drew=oracle(logits,o,history if group==0 else [],seed=seed,subseq=subseq)
            calls.append({"frame":frame,"codebook":group,"logits":logits,"options":o,"history":list(history if group==0 else []),"suppress_from":None,"allow_token":None,"expected":token,"subsequence":subseq})
            row.append(token)
            if drew: subseq += 1
            if group==0: history.append(token)
        frames.append(row)
    return {"id":cid,"seed":seed,"talker_do_sample":talker,"subtalker_do_sample":cp,"frames":frames,"c0_history":history,"cp_history_empty":True,"calls":calls,"draw_count":subseq}

def build():
    cases=[]
    base={"temperature":1.0,"top_k":4,"top_p":1.0,"repetition_penalty":1.0}
    for cid,t,c,s in [("greedy_greedy",False,False,17),("greedy_sampled",False,True,19),("sampled_greedy",True,False,23),("sampled_sampled",True,True,29)]: cases.append(make_case(cid,t,c,s,base))
    # Explicit boundary and signed repetition cases are independent captured logits.
    cases.append(make_case("top_k_boundary",True,False,31,{"temperature":1.0,"top_k":2,"top_p":1.0,"repetition_penalty":1.0}))
    cases.append(make_case("top_p_boundary",True,False,37,{"temperature":1.0,"top_k":0,"top_p":0.55,"repetition_penalty":1.0}))
    logits=[-2.0,-1.0,4.0,3.0,-10.0,2.0,1.0,0.0]
    opts={"do_sample":True,"temperature":1.0,"top_k":0,"top_p":1.0,"repetition_penalty":1.5}
    tok,_=oracle(logits,opts,[0,1],seed=41,subseq=0)
    cases.append({"id":"repetition_positive_negative","seed":41,"talker_do_sample":True,"subtalker_do_sample":False,"frames":[[tok]],"c0_history":[tok],"cp_history_empty":True,"draw_count":1,"calls":[{"frame":0,"codebook":0,"logits":logits,"options":opts,"history":[0,1],"suppress_from":None,"allow_token":None,"expected":tok,"subsequence":0}]})
    tie_opts={"do_sample":True,"temperature":1.0,"top_k":2,"top_p":1.0,"repetition_penalty":1.0}
    tie_logits=[4.0,3.0,3.0,-10.0]; tie,_=oracle(tie_logits,tie_opts,[],seed=43,subseq=0)
    cases.append({"id":"top_k_tie_boundary","seed":43,"talker_do_sample":True,"subtalker_do_sample":False,"frames":[[tie]],"c0_history":[tie],"cp_history_empty":True,"draw_count":1,"calls":[{"frame":0,"codebook":0,"logits":tie_logits,"options":tie_opts,"history":[],"suppress_from":None,"allow_token":None,"expected":tie,"subsequence":0}]})
    order_logits=[f32(-0.0001*i) for i in range(64)]
    order_opts={"do_sample":True,"temperature":1.0,"top_k":0,"top_p":1.0,"repetition_penalty":1.0}
    order_tok,_=oracle(order_logits,order_opts,[],seed=47,subseq=0)
    cases.append({"id":"original_vocab_sum_order_boundary","seed":47,"talker_do_sample":True,"subtalker_do_sample":False,"frames":[[order_tok]],"c0_history":[order_tok],"cp_history_empty":True,"draw_count":1,"calls":[{"frame":0,"codebook":0,"logits":order_logits,"options":order_opts,"history":[],"suppress_from":None,"allow_token":None,"expected":order_tok,"subsequence":0}]})
    cutoff_logits=[8.0, -7.0, -7.1, -23.0, -30.0]
    cutoff_opts={"do_sample":True,"temperature":1.0,"top_k":0,"top_p":0.9,"repetition_penalty":1.0}
    cutoff_tok,_=oracle(cutoff_logits,cutoff_opts,[],seed=49,subseq=0)
    cases.append({"id":"top_p_relative_cutoff_boundary","seed":49,"talker_do_sample":True,"subtalker_do_sample":False,"frames":[[cutoff_tok]],"c0_history":[cutoff_tok],"cp_history_empty":True,"draw_count":1,"calls":[{"frame":0,"codebook":0,"logits":cutoff_logits,"options":cutoff_opts,"history":[],"suppress_from":None,"allow_token":None,"expected":cutoff_tok,"subsequence":0}]})
    eos_opts={"do_sample":False,"temperature":0.0,"top_k":0,"top_p":1.0,"repetition_penalty":1.0}
    eos_calls=[]; eos_frames=[]
    for frame in range(3):
        logits=[-4.0]*2048; logits[3+frame]=2.0; logits[2047]=10.0
        allow=2047 if frame>=2 else None
        tok,_=oracle(logits,eos_opts,[],suppress_from=1024,allow=allow,seed=53,subseq=0)
        eos_calls.append({"frame":frame,"codebook":0,"logits":logits,"options":eos_opts,"history":[],"suppress_from":1024,"allow_token":allow,"expected":tok,"subsequence":0})
        eos_frames.append([tok])
    # CP is deliberately unsuppressed: a reserved-suffix id remains selectable.
    cp_logits=[-2.0]*2048; cp_logits[1500]=5.0
    eos_calls.append({"frame":3,"codebook":1,"logits":cp_logits,"options":eos_opts,"history":[],"suppress_from":None,"allow_token":None,"expected":1500,"subsequence":0})
    cases.append({"id":"talker_suppression_eos_truncation","seed":53,"talker_do_sample":False,"subtalker_do_sample":False,"frames":eos_frames,"c0_history":[r[0] for r in eos_frames],"cp_history_empty":True,"draw_count":0,"calls":eos_calls})
    # Full-width sequence with terminal c0 EOS-only step.
    seq_calls=[]; seq_frames=[]; seq_hist=[]
    for frame in range(2):
        row=[]
        for group in range(16):
            logits=[-4.0]*2048; token=1500 if group==1 else (10+frame*16+group) % 1024; logits[token]=5.0
            call={"frame":frame,"codebook":group,"logits":logits,"options":eos_opts,"history":list(seq_hist) if group==0 else [],"suppress_from":1024 if group==0 else None,"allow_token":None,"expected":token,"subsequence":0}
            seq_calls.append(call); row.append(token)
            if group==0: seq_hist.append(token)
        seq_frames.append(row)
    logits=[-4.0]*2048; logits[2047]=9.0
    seq_calls.append({"frame":2,"codebook":0,"logits":logits,"options":eos_opts,"history":list(seq_hist),"suppress_from":1024,"allow_token":2047,"expected":2047,"subsequence":0})
    cases.append({"id":"talker_suppression_eos_sequence","seed":59,"talker_do_sample":False,"subtalker_do_sample":False,"frames":seq_frames,"c0_history":seq_hist,"cp_history_empty":True,"draw_count":0,"calls":seq_calls})
    for c in cases:
        c.setdefault("kind", "sequence")
        if c["id"] in {"repetition_positive_negative", "top_k_tie_boundary", "talker_suppression_eos_truncation"}:
            c["kind"] = "operator_vector"
    return {"version":2,"fixture_id":"p02-deterministic-token-sequences","path":"fixtures/alignment/p02_deterministic_token_sequences.json","source_revision":REV,"official_qwen_revision":QWEN_REV,"generated_by":"tools/generate_deterministic_token_sequences.py independent oracle","command":"python tools/generate_deterministic_token_sequences.py --check fixtures/alignment/p02_deterministic_token_sequences.json","sampling_contract":"qwentts.cpp Philox4x32-10; explicit captured logits/options/history; P03 owns model-logit parity","cases":cases}

def validate_payload(current, expected):
    for k in expected:
        if k != "cases" and current.get(k) != expected[k]:
            raise RuntimeError(f"FIXTURE_MISMATCH: {k}")
    if current.get("cases") != expected["cases"]:
        raise RuntimeError("FIXTURE_MISMATCH: cases/oracle")
    ids=[c["id"] for c in current["cases"]]
    if len(ids)!=len(set(ids)) or len(ids)<7:
        raise RuntimeError("FIXTURE_MISMATCH: ids/coverage")

def validate(path):
    current=json.loads(path.read_text(encoding="utf-8")); expected=build(); validate_payload(current, expected)
    manifest=json.loads((ROOT/"config/fixtures.json").read_text(encoding="utf-8"))
    validate_manifest_sha(path, manifest, "p02-deterministic-token-sequences", "FIXTURE_MISMATCH: manifest SHA")
    validate_real_payload(ROOT/"fixtures/alignment/p02_deterministic_token_sequences_real.json", manifest)

def validate_manifest_sha(path, manifest, fixture_id, error):
    digest=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    entries=[e for e in manifest.get("fixtures",[]) if e.get("id")==fixture_id]
    if len(entries)!=1 or entries[0].get("sha256")!=digest: raise RuntimeError(error)

def validate_real_payload(path, manifest, data_override=None):
    data=data_override if data_override is not None else json.loads(Path(path).read_text(encoding="utf-8"))
    required={"snapshot_revision":"5d83992436eae1d760afd27aff78a71d676296fc","official_qwen_revision":QWEN_REV,"qwentts_cpp_revision":REV}
    for key,value in required.items():
        if data.get(key)!=value: raise RuntimeError(f"REAL_FIXTURE_MISMATCH: {key}")
    if not isinstance(data.get("frames"),list) or len(data["frames"])<2 or any(len(r)!=16 for r in data["frames"]):
        raise RuntimeError("REAL_FIXTURE_MISMATCH: frames")
    digest=hashlib.sha256(Path(path).read_bytes()).hexdigest() if data_override is None else hashlib.sha256(json.dumps(data_override,separators=(',',':')).encode()).hexdigest()
    entries=[e for e in manifest.get("fixtures",[]) if e.get("id")=="p02-deterministic-token-sequences-real"]
    if len(entries)!=1 or entries[0].get("sha256")!=digest: raise RuntimeError("REAL_FIXTURE_MISMATCH: manifest SHA")

def self_test():
    expected=build(); base=copy.deepcopy(expected)
    mutations=[]
    for label, mutate in [
        ("duplicate", lambda x: x["cases"].append(copy.deepcopy(x["cases"][0]))),
        ("missing", lambda x: x["cases"].pop()),
        ("revision", lambda x: x.__setitem__("source_revision", "bad")),
        ("token", lambda x: x["cases"][0]["calls"][0].__setitem__("expected", 999)),
        ("matrix", lambda x: x["cases"][0]["frames"][0].__setitem__(0, 999)),
    ]:
        candidate=copy.deepcopy(base); mutate(candidate)
        try: validate_payload(candidate, expected)
        except RuntimeError: continue
        raise RuntimeError(f"SELF_TEST_ACCEPTED: {label}")
    manifest=json.loads((ROOT/"config/fixtures.json").read_text(encoding="utf-8"))
    stale_synthetic=copy.deepcopy(manifest)
    next(e for e in stale_synthetic["fixtures"] if e.get("id")=="p02-deterministic-token-sequences")["sha256"]="stale"
    try: validate_manifest_sha(OUT, stale_synthetic, "p02-deterministic-token-sequences", "synthetic stale SHA")
    except RuntimeError: pass
    else: raise RuntimeError("SELF_TEST_ACCEPTED: synthetic manifest SHA mutation")
    real_path=ROOT/"fixtures/alignment/p02_deterministic_token_sequences_real.json"
    real=json.loads(real_path.read_text(encoding="utf-8")); validate_real_payload(real_path, manifest)
    required={"snapshot_revision":"5d83992436eae1d760afd27aff78a71d676296fc","official_qwen_revision":QWEN_REV,"qwentts_cpp_revision":REV}
    for key, value in required.items():
        candidate=copy.deepcopy(real); candidate[key]="bad"
        try: validate_real_payload(real_path, manifest, candidate)
        except RuntimeError: continue
        raise RuntimeError(f"SELF_TEST_ACCEPTED: real {key}")
    stale_real=copy.deepcopy(manifest)
    next(e for e in stale_real["fixtures"] if e.get("id")=="p02-deterministic-token-sequences-real")["sha256"]="stale"
    try: validate_real_payload(real_path, stale_real)
    except RuntimeError: pass
    else: raise RuntimeError("SELF_TEST_ACCEPTED: real manifest SHA mutation")
    bad_manifest=copy.deepcopy(manifest); bad_manifest["fixtures"]=bad_manifest["fixtures"]+[copy.deepcopy(next(e for e in bad_manifest["fixtures"] if e.get("id")=="p02-deterministic-token-sequences-real"))]
    try: validate_real_payload(real_path,bad_manifest)
    except RuntimeError: pass
    else: raise RuntimeError("SELF_TEST_ACCEPTED: real manifest duplicate")
    print("DETERMINISTIC_SEQUENCE_SELF_TEST_OK")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--check",nargs="?",const=OUT,type=Path); p.add_argument("--self-test",action="store_true"); p.add_argument("--output",type=Path,default=OUT); a=p.parse_args()
    if a.self_test: self_test(); return
    if a.check: validate(a.check); print(f"DETERMINISTIC_SEQUENCE_CHECK_OK {a.check}")
    else: a.output.write_text(json.dumps(build(),indent=2)+"\n",encoding="utf-8"); print(f"DETERMINISTIC_SEQUENCE_GENERATED {a.output}")
if __name__=="__main__": main()
