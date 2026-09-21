$log = 'C:\Users\china\elevated_install.log'
function Say($m) { Write-Host $m; Add-Content -Path $log -Value $m }
$isAdmin = ([Security.Principal.WindowsPrincipal]`
  [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(`
  [Security.Principal.WindowsBuiltInRole]::Administrator)
Say ("worker installer, admin=" + $isAdmin)

$t1 = New-ScheduledTaskTrigger -AtLogOn
$t2 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
        -RepetitionInterval (New-TimeSpan -Minutes 1) -RepetitionDuration ([TimeSpan]::MaxValue)
$set = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
$a = New-ScheduledTaskAction -Execute 'powershell.exe' `
      -Argument '-NoProfile -ExecutionPolicy Bypass -File "C:\Users\china\autoforge_node\elevated_drain.ps1"'
try {
  Register-ScheduledTask -TaskName 'autoforge-elevated-worker' -Action $a -Trigger $t1,$t2 `
    -Settings $set -RunLevel Highest -Force | Out-Null
  Say 'registered: autoforge-elevated-worker (at logon + every 1 minute, highest)'
} catch { Say ('FAILED: ' + $_.Exception.Message) }

# prove the channel end to end, now, in this elevated session
$probe = Join-Path 'C:\Users\china\autoforge_node\elevated_queue\in' 'selftest.json'
@{ cmd = 'whoami /groups | findstr /C:"Mandatory Level"' } | ConvertTo-Json | Set-Content $probe -Encoding UTF8
& 'powershell.exe' -NoProfile -ExecutionPolicy Bypass -File 'C:\Users\china\autoforge_node\elevated_drain.ps1'
$r = Join-Path 'C:\Users\china\autoforge_node\elevated_queue\out' 'selftest.json'
if (Test-Path $r) { Say ('selftest result: ' + ((Get-Content $r -Raw) -replace '\s+', ' ')) }
else { Say 'selftest produced no result' }
Start-Sleep -Seconds 2
