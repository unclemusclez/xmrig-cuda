import struct

data = open('gpu_program.bin','rb').read()
prog = struct.unpack('<256I', data)

op_names = {
    0:'IADD_RS', 1:'IADD_M', 2:'ISUB_R', 3:'ISUB_M',
    4:'IMUL_R', 5:'IMUL_M', 6:'IMULH_R', 7:'IMULH_M',
    8:'ISMULH_R', 9:'ISMULH_M', 10:'IMUL_RCP', 11:'INEG_R',
    12:'IXOR_R', 13:'IXOR_M', 14:'IROR_R', 15:'IROL_R',
    16:'ISWAP_R', 17:'FSWAP_R', 18:'FADD_R', 19:'FADD_M',
    20:'FSUB_R', 21:'FSUB_M', 22:'FSCAL_R', 23:'FMUL_R',
    24:'FDIV_M', 25:'FSQRT_R', 26:'CBRANCH', 27:'CFROUND',
    28:'ISTORE', 29:'NOP'
}

seq = []
for ip, inst in enumerate(prog):
    opcode = (inst >> 20) & 15
    location = (inst >> 14) & 1
    num_insts = ((inst >> 24) & 15) + 1
    num_fp = ((inst >> 28) & 15)
    
    dst = (inst >> 0) & 7
    src = (inst >> 3) & 7
    imm_off = (inst >> 6) & 255
    
    for w in range(num_insts):
        is_fp = w < num_fp
        if is_fp:
            actual_ip = ip + (w // 2)
            sub = w % 2
            src_w = (inst >> 4) & 7
        else:
            actual_ip = ip + (w - num_fp)
            sub = w - num_fp
            src_w = (inst >> 3) & 7
        
        op = op_names.get(opcode, 'UNK%d' % opcode)
        seq.append((actual_ip, w, sub, is_fp, opcode, op, dst, src_w, (inst >> 14) & 1, (inst >> 6) & 255))

print('Expanded GPU program: %d sequential instructions' % len(seq))
for i, s in enumerate(seq[:100]):
    print('  [%3d] ip=%3d w=%d sub=%d fp=%s op=%2d %-10s dst=%d src=%d loc=%d imm_off=%3d' % (i, s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9]))