import hashlib
# base input = bytes 0..75
base = bytearray(range(76))
# embed_nonce(0): bytes 39..42 = nonce(0) LE bytes = all zero
base[39]=0; base[40]=0; base[41]=0; base[42]=0
h = hashlib.blake2b(bytes(base), digest_size=64).hexdigest()
print("CPU blake2b nonce0 :", h)
import os
if os.path.exists("gpu_firsthash.bin"):
    d = open("gpu_firsthash.bin","rb").read()
    print("GPU firsthash size :", len(d))
    # gpu_firsthash.bin is 8 uint64 LE = 64 bytes
    print("GPU firsthash      :", d.hex())
    print("MATCH" if d.hex()==h else "MISMATCH")
else:
    print("no gpu_firsthash.bin in cwd")
