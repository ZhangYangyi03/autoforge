#  Elevated setup for the autoforge node.  Written 2026-09-21.
#  Right-click install_elevated.cmd -> Run as administrator.
$ErrorActionPreference = 'Continue'
$log = 'C:\Users\china\elevated_install.log'
function Say($m) { Write-Host $m; Add-Content -Path $log -Value $m }
Set-Content -Path $log -Value ("==== elevated run " + (Get-Date) + " ====")

Say ("whoami: " + (whoami))
Say ("admin : " + ([Security.Principal.WindowsPrincipal]`
      [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(`
      [Security.Principal.WindowsBuiltInRole]::Administrator))

# ---- 1. KOS's ssh key -------------------------------------------------------
Say ""
Say "== 1/3  KOS ssh key =="
$key = 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMUygCOdyir6BfcWAyPssqmz06nv3Z/fBP1fKpVTcmk4 KOS@kylin'
$dir = 'C:\ProgramData\ssh'
$kf  = Join-Path $dir 'administrators_authorized_keys'
if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
$cur = ''
if (Test-Path $kf) { try { $cur = Get-Content $kf -Raw } catch { $cur = '' } }
Say ("existing size: " + $cur.Length)
if ($cur -notmatch 'KOS@kylin') {
  # write as UTF8 with no BOM; sshd rejects a BOM as a malformed key
  $out = $cur.TrimEnd() + "`r`n" + $key + "`r`n"
  [IO.File]::WriteAllText($kf, $out, (New-Object Text.UTF8Encoding($false)))
  Say "key appended"
} else { Say "key already present" }
# sshd refuses the file if anyone but Administrators/SYSTEM can write it
icacls $kf /inheritance:r /grant 'Administrators:F' /grant 'SYSTEM:F' | Out-Null
Say "acl: Administrators + SYSTEM only"
$svc = Get-Service sshd -ErrorAction SilentlyContinue
if ($svc) {
  Say ("sshd: " + $svc.Status)
  if ($svc.Status -ne 'Running') { Start-Service sshd; Say "sshd started" }
} else { Say "sshd: no such service" }

# ---- 2. the two services as self-restarting tasks ---------------------------
Say ""
Say "== 2/3  scheduled tasks =="
$trig = New-ScheduledTaskTrigger -AtLogOn
$set  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
        -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
$pairs = @(
  @{ n='autoforge-market'; a=(New-ScheduledTaskAction -Execute 'cmd.exe' -Argument '/c "C:\Users\china\start_market.cmd"') },
  @{ n='autoforge-node';   a=(New-ScheduledTaskAction -Execute 'cmd.exe' -Argument '/c "C:\Users\china\autoforge_node\start_node.cmd"') }
)
foreach ($p in $pairs) {
  try {
    Register-ScheduledTask -TaskName $p.n -Action $p.a -Trigger $trig -Settings $set -RunLevel Highest -Force | Out-Null
    Say ("registered: " + $p.n)
  } catch { Say ("FAILED to register " + $p.n + ": " + $_.Exception.Message) }
}

# ---- 3. start them now ------------------------------------------------------
Say ""
Say "== 3/3  starting =="
foreach ($n in 'autoforge-market','autoforge-node') {
  $o = schtasks /run /tn $n 2>&1
  Say ("run " + $n + " -> " + $o)
}
Start-Sleep -Seconds 10

# ---- result -----------------------------------------------------------------
Say ""
Say "---- result ----"
foreach ($port in 8000,8077) {
  $path = if ($port -eq 8077) { '/node/health' } else { '/health' }
  $url  = 'http://127.0.0.1:' + $port + $path
  try {
    $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 8 $url
    Say ("port " + $port + " UP   " + $r.Content.Substring(0, [Math]::Min(80, $r.Content.Length)))
  } catch { Say ("port " + $port + " DOWN " + $_.Exception.Message) }
}
Say "log: $log"
