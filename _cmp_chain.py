import os, sys
def rd(p):
    return open(p,'rb').read() if os.path.exists(p) else None
def cmp(a,b,tag):
    da, db = rd(a), rd(b)
    if da is None or db is None:
        print(f"{tag}: MISSING ({'cpu' if da is None else 'gpu'})"); return False
    ok = da == db
    print(f"{tag}: {'MATCH' if ok else 'MISMATCH'} ({len(da)} vs {len(db)} bytes)")
    return ok

cpu_chain = "cpu_chain_h0_input.bin"
cmp(None if not os.path.exists(cpu_chain) else cpu_chain, "gpu_firsthash.bin", "h0_input(first hash)")
cmp("cpu_chain_h0_evolved.bin", "gpu_chain_h0.bin", "h0_evolved(after fillAes1Rx4)")
for i in range(8):
    cmp(f"cpu_scratch_p{i}.bin", f"gpu_scratch_p{i}.bin", f"scratch before prog{i}")
    # entropy: first 128 bytes of cpu Program vs gpu entropy first 128
    cp = rd(f"cpu_prog_p{i}.bin"); ge = rd(f"gpu_entropy_p{i}.bin")
    if cp is not None and ge is not None:
        print(f"entropy prog{i}: {'MATCH' if cp[:128]==ge[:128] else 'MISMATCH'}")
        print(f"program prog{i}: {'MATCH' if cp[len(cp)-2048:]==ge[128:128+2048] else 'MISMATCH'} (cpu_prog {len(cp)}B, gpu_ent {len(ge)}B)")
    cmp(f"cpu_rf_p{i}.bin", f"gpu_rf_p{i}.bin", f"regfile after prog{i}")
    if i < 7:
        cmp(f"cpu_chain_h{i+1}.bin", f"gpu_chain_h{i+1}.bin", f"chain hash h{i+1}")
