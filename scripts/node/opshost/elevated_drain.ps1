#  autoforge elevated worker -- drains a queue of privileged requests.
#
#  Why a queue and not a listener: a root process holding an open port is an
#  unauthenticated remote-execution service, and this machine already has one
#  public tunnel on it. A directory is the same capability with a much smaller
#  surface: the requester needs write access to one folder, and the auditor can
#  read every request and every result afterwards.
#
#  Honest about what this is: WHOEVER CAN WRITE TO elevated_queue\in CAN RUN CODE
#  AS THIS TASK'S USER. That directory's ACL is therefore the whole security
#  model. It is set to this user + Administrators only, and that is written down
#  here because a design whose security rests on an ACL should say which ACL.
$ErrorActionPreference = 'Continue'
$root = 'C:\Users\china\autoforge_node\elevated_queue'
$in   = Join-Path $root 'in'
$out  = Join-Path $root 'out'
$done = Join-Path $root 'done'
foreach ($d in @($in, $out, $done)) { if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null } }

$isAdmin = ([Security.Principal.WindowsPrincipal]`
  [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(`
  [Security.Principal.WindowsBuiltInRole]::Administrator)

Get-ChildItem -Path $in -Filter '*.json' -ErrorAction SilentlyContinue | ForEach-Object {
  $req = $null
  try { $req = Get-Content $_.FullName -Raw | ConvertFrom-Json } catch { }
  $result = @{ id = $_.BaseName; ran_as = (whoami); admin = $isAdmin; ts = (Get-Date).ToString('s') }
  if ($req -and $req.cmd) {
    $stdout = ''; $stderr = ''; $code = -1
    try {
      $tmpO = [IO.Path]::GetTempFileName(); $tmpE = [IO.Path]::GetTempFileName()
      $p = Start-Process -FilePath 'cmd.exe' -ArgumentList ('/c ' + $req.cmd) -NoNewWindow -Wait `
            -RedirectStandardOutput $tmpO -RedirectStandardError $tmpE -PassThru
      $code = $p.ExitCode
      $stdout = (Get-Content $tmpO -Raw); $stderr = (Get-Content $tmpE -Raw)
      Remove-Item $tmpO, $tmpE -Force -ErrorAction SilentlyContinue
    } catch { $stderr = $_.Exception.Message }
    $result.exit = $code
    $result.stdout = $stdout
    $result.stderr = $stderr
  } else { $result.exit = -2; $result.stderr = 'no cmd in request' }
  $result | ConvertTo-Json -Depth 6 | Set-Content -Path (Join-Path $out ($_.BaseName + '.json')) -Encoding UTF8
  Move-Item $_.FullName (Join-Path $done $_.Name) -Force
}
