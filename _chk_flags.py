import struct
d = open("gpu_vmstate_p0.bin","rb").read()
print("vmstate size", len(d))
prog_len = struct.unpack("<I", d[160:164])[0]
print("program_length =", prog_len)
# compiled_program at byte 1024
words = []
for w in range(256):
    off = 1024 + w*4
    if off+4 <= 2048:
        words.append(struct.unpack("<I", d[off:off+4])[0])
# group_flags at byte 2048 (64 bytes = 16 uint32)
flags = []
for j in range(16):
    off = 2048 + j*4
    flags.append(struct.unpack("<I", d[off:off+4])[0])
def flag_bits(w):
    return (flags[w>>4] >> ((w&15)*2)) & 3
bad = 0
for w in range(prog_len):
    word = words[w]
    op = (word >> 20) & 15
    is_cb = (op == 9)
    is_cf = (op == 13)
    fb = flag_bits(w)
    expect = (1 if is_cb else 0) | (2 if is_cf else 0)
    if fb != expect:
        bad += 1
        if bad <= 10:
            print(f"word[{w}] op={op} word={word:08x} flag={fb} expect={expect}  (cb={is_cb} cf={is_cf})")
print(f"flag/opcode consistency: {bad} inconsistencies out of {prog_len} words")
# also count control words
ncb = sum(1 for w in range(prog_len) if ((words[w]>>20)&15)==9)
ncf = sum(1 for w in range(prog_len) if ((words[w]>>20)&15)==13)
print(f"CBRANCH words={ncb}, CFROUND words={ncf}")
