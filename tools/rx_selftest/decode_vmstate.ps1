$ErrorActionPreference = "Stop"
function DecodeVM($path) {
    $b = [IO.File]::ReadAllBytes($path)
    # control block at uint64 R[16] = byte offset 128
    $ma   = [BitConverter]::ToUInt32($b, 128)
    $mx   = [BitConverter]::ToUInt32($b, 132)
    $ar   = [BitConverter]::ToUInt32($b, 136)
    $dso  = [BitConverter]::ToUInt32($b, 140)
    $emask0 = [BitConverter]::ToUInt64($b, 144)
    $emask1 = [BitConverter]::ToUInt64($b, 152)
    $plen = [BitConverter]::ToUInt32($b, 160)
    $obj = [ordered]@{ ma=$ma; mx=$mx; addrRegs=$ar; dsOff=$dso; eMask0=$emask0; eMask1=$emask1; progLen=$plen }
    # compiled program at byte 1024, 256 words
    $words = @()
    for ($i = 0; $i -lt 256; $i++) { $words += [BitConverter]::ToUInt32($b, 1024 + $i*4) }
    $obj.words = $words
    return $obj
}
$p0 = DecodeVM "gpu_vmstate_p0.bin"
$p1 = DecodeVM "gpu_vmstate_p1.bin"
$p2 = DecodeVM "gpu_vmstate_p2.bin"
foreach ($n in @("p0","p1","p2")) {
    $p = Get-Variable $n -ValueOnly
    Write-Host ("{0}: ma={1:x8} mx={2:x8} addrRegs={3:x8} dsOff={4:x8} eMask0={5:x16} eMask1={6:x16} progLen={7}" -f $n, $p.ma, $p.mx, $p.addrRegs, $p.dsOff, $p.eMask0, $p.eMask1, $p.progLen)
}
# opcode histogram + group-header walk for each
function WalkProg($p, $name) {
    $w = $p.words
    $opcodes = @{}
    $ip = 0
    $groups = 0
    while ($ip -lt $p.progLen) {
        $hdr = $w[$ip]
        $nw = ($hdr -shr 24) -band 7
        $nf = ($hdr -shr 28) -band 7
        $groups++
        # count opcodes in this group's words (header word + integer/fp slots)
        $ipnext = $ip + ($nw - $nf) + 1
        for ($k = $ip; $k -lt $ipnext; $k++) {
            $op = ($w[$k] -shr 20) -band 15
            if ($opcodes.ContainsKey($op)) { $opcodes[$op]++ } else { $opcodes[$op] = 1 }
        }
        $ip = $ipnext
    }
    $line = ($opcodes.GetEnumerator() | Sort-Object Name | ForEach-Object { "op$($_.Name)x$($_.Value)" }) -join " "
    Write-Host ("{0}: groups={1} {2}" -f $name, $groups, $line)
}
WalkProg $p0 "p0"
WalkProg $p1 "p1"
WalkProg $p2 "p2"
