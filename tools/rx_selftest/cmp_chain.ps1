$ErrorActionPreference = "Stop"
function CmpFiles($a, $b) {
    $x = [IO.File]::ReadAllBytes($a); $y = [IO.File]::ReadAllBytes($b)
    if ($x.Length -ne $y.Length) { return "SIZE" }
    for ($i = 0; $i -lt $x.Length; $i++) { if ($x[$i] -ne $y[$i]) { return "DIFF@$i" } }
    return "OK"
}
Write-Host ("h0  gpu vs cpu_evolved : " + (CmpFiles "gpu_chain_h0.bin" "cpu_chain_h0_evolved.bin"))
Write-Host ("h0i gpu vs cpu_input   : " + (CmpFiles "gpu_chain_h0.bin" "cpu_chain_h0_input.bin"))
for ($i = 1; $i -le 7; $i++) {
    Write-Host ("h$i  gpu vs cpu        : " + (CmpFiles "gpu_chain_h$i.bin" "cpu_chain_h$i.bin"))
}
for ($i = 0; $i -le 7; $i++) {
    Write-Host ("rf$i gpu vs cpu        : " + (CmpFiles "gpu_rf_p$i.bin" "cpu_rf_p$i.bin"))
}
$fh = [BitConverter]::ToString([IO.File]::ReadAllBytes("cpu_final_hash.bin")).Replace("-", "").ToLower()
$gh = ((Get-Content "gpu_out.txt")[0] -split " ")[1]
Write-Host "final cpu: $fh"
Write-Host "final gpu: $gh"
