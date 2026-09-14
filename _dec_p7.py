import struct
d = open("gpu_vmstate_p7.bin","rb").read()
print("size", len(d))
prog_len = struct.unpack("<I", d[160:164])[0]
print("program_length", prog_len)
def decode(w):
    word = struct.unpack("<I", d[1024+w*4:1024+w*4+4])[0]
    op = (word>>20)&15
    dstf = word&7
    srcf = (word>>3)&7
    immf = (word>>6)&255
    locf = (word>>14)&1
    nw = (word>>24)&7
    nfp = (word>>28)&7
    names={0:"IADD_RS",1:"IADD_M",2:"ISUB",3:"IMUL?"}
    # packed opcodes per inner_loop: 0,1=add;2=mul;3=xor;4=ismulh;5=ineg;6=imulh;7=ror/rol;8=iswap;9=cbranch;10=istore;11=fswap;12=fp fma;13=cfround;14=fsqrt;15=fdiv
    nm={0:"ADD",1:"ADD",2:"IMUL",3:"IXOR",4:"ISMULH",5:"INEG",6:"IMULH",7:"IROR/IROL",8:"ISWAP",9:"CBRANCH",10:"ISTORE",11:"FSWAP",12:"FP-FMA",13:"CFROUND",14:"FSQRT",15:"FDIV"}[op]
    print(f"word[{w}] = {word:08x} op={op}({nm}) dst={dstf} src={srcf} imm_field={immf} loc={locf} num_workers(enc)={nw} num_fp={nfp}")
for w in range(8):
    decode(w)
print("--- R registers (bytes 0..63) ---")
for r in range(8):
    v = struct.unpack("<Q", d[r*8:r*8+8])[0]
    print(f"  R{r} = {v:016x}")
