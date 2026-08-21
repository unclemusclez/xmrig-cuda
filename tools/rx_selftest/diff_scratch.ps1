$ErrorActionPreference = "Stop"
$x = [IO.File]::ReadAllBytes("gpu_scratch_p2.bin")
$y = [IO.File]::ReadAllBytes("cpu_scratch_p2.bin")
$diffs = @()
for ($i = 0; $i -lt $x.Length; $i++) { if ($x[$i] -ne $y[$i]) { $diffs += $i } }
Write-Host "total differing bytes: $($diffs.Count)"
Write-Host "first diffs:"
$diffs | Select-Object -First 12 | ForEach-Object { Write-Host ("  offset {0} (0x{0:X})" -f $_) }
# group into contiguous runs
if ($diffs.Count -gt 0) {
    $runs = @(); $start = $diffs[0]; $prev = $diffs[0]
    for ($i = 1; $i -lt $diffs.Count; $i++) {
        if ($diffs[$i] -eq $prev + 1) { $prev = $diffs[$i] }
        else { $runs += ,@($start, $prev); $start = $diffs[$i]; $prev = $diffs[$i] }
    }
    $runs += ,@($start, $prev)
    Write-Host "contiguous diff runs: $($runs.Count)"
    $runs | Select-Object -First 8 | ForEach-Object { Write-Host ("  run {0}..{1} (len {2})" -f $_[0], $_[1], ($_[1]-$_[0]+1)) }
}
