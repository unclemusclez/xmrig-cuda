$ErrorActionPreference = "Continue"
$root = "P:\GitRepos\xmrig-cuda"

# Optional first arg: bfactor to test on the GPU side (default 6).
$bfactor = if ($args.Count -ge 1) { $args[0] } else { 6 }

# Canonical test vector: 32-byte zero seed, 76-byte base input (bytes 0..75).
$seed = "00" * 32
$base = -join (0..75 | ForEach-Object { "{0:x2}" -f $_ })
$count = 32

$log = "$root\tools\rx_selftest\run_test.log"
$env:PATH = "C:\Program Files\AMD\ROCm\7.2\bin;$env:PATH"

# Native tools print progress to stderr; redirect so it can't throw and pollute the
# comparison stream (which must contain only "<idx> <hash>" lines on stdout).
Write-Host "=== CPU oracle #1 (golden, oracle.cpp) ==="
$golden = & "$root\tools\rx_selftest\oracle.exe" $seed $base $count 2>> $log
$golden | ForEach-Object { Write-Host "  $_" }

Write-Host "=== CPU oracle #2 (monero rx-slow-hash.c port) ==="
$monero = & "$root\tools\rx_selftest\monero_oracle.exe" $seed $base $count 2>> $log
$monero | ForEach-Object { Write-Host "  $_" }

Write-Host "=== GPU selftest (bfactor=$bfactor) ==="
$gpu = & "$root\tools\rx_selftest\selftest.exe" $seed $base $count $bfactor 2>> $log
$gpu | ForEach-Object { Write-Host "  $_" }

Write-Host "=== Comparison ==="
$cpuMismatch = 0
$gpuMismatch = 0
for ($i = 0; $i -lt $count; $i++) {
    $g = $golden[$i].Split(' ')[1]
    $m = $monero[$i].Split(' ')[1]
    $u = $gpu[$i].Split(' ')[1]
    if ($g -ne $m) {
        $cpuMismatch++
        Write-Host "  nonce $i : CPU-ORACLES-DISAGREE (orchestration bug in oracle.cpp?)"
        Write-Host "     oracle.cpp=$g"
        Write-Host "     monero   =$m"
    }
    if ($g -eq $u) {
        Write-Host "  nonce $i : GPU MATCH (cpu=$g)"
    } else {
        $gpuMismatch++
        Write-Host "  nonce $i : GPU MISMATCH"
        Write-Host "     cpu=$g"
        Write-Host "     gpu=$u"
    }
}
Write-Host "=== CPU-oracle agreement: $cpuMismatch disagreement(s) out of $count ==="
Write-Host "=== GPU result: $gpuMismatch mismatch(es) out of $count ==="
if ($cpuMismatch -ne 0) { Write-Host "CPU-ORACLE MISMATCH -- fix oracle.cpp orchestration before trusting GPU result" }
if ($gpuMismatch -eq 0) { Write-Host "SELF-TEST PASSED" } else { Write-Host "SELF-TEST FAILED" }
exit $gpuMismatch
