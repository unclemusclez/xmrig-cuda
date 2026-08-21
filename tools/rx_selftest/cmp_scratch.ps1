$ErrorActionPreference = "Stop"
function FirstDiff($a, $b) {
    $x = [IO.File]::ReadAllBytes($a); $y = [IO.File]::ReadAllBytes($b)
    if ($x.Length -ne $y.Length) { return -2 }
    for ($i = 0; $i -lt $x.Length; $i++) { if ($x[$i] -ne $y[$i]) { return $i } }
    return -1
}
for ($i = 0; $i -le 7; $i++) {
    $d = FirstDiff "gpu_scratch_p$i.bin" "cpu_scratch_p$i.bin"
    if ($d -eq -1) { Write-Host "scratch_p$i : IDENTICAL" }
    elseif ($d -eq -2) { Write-Host "scratch_p$i : SIZE MISMATCH" }
    else { Write-Host "scratch_p$i : first diff @ $d" }
}
# compiled-program region of vmstates (bytes 1024..2047)
function FirstDiffRegion($a, $b, $off, $len) {
    $x = [IO.File]::ReadAllBytes($a); $y = [IO.File]::ReadAllBytes($b)
    for ($i = 0; $i -lt $len; $i++) { if ($x[$off + $i] -ne $y[$off + $i]) { return $i } }
    return -1
}
Write-Host ""
Write-Host "done"
