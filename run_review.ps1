# Entry point of the optimal-combination review. The scheduled task Review-OptimalCombo runs it on weekdays at
# 15:10 (review) and 16:40 (retry, or data check when the review is done); it can also be run by hand:
#   powershell -ExecutionPolicy Bypass -File D:\review\run_review.ps1 -Show                   # today, progress shown, page opens
#   powershell -ExecutionPolicy Bypass -File D:\review\run_review.ps1 -Show -Date 2026-10-14  # replay a past day
# (D:\review\立即复盘.cmd does the first line with a double-click.)
param([string]$Date = '', [switch]$Full, [switch]$Show)
$ErrorActionPreference = 'Continue'
$root = $PSScriptRoot
$logs = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$lock = Join-Path $logs 'run.lock'
function Write-Log($msg) {
  Add-Content -Path (Join-Path $logs 'scheduler.log') -Value ('[{0}] {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg) -Encoding UTF8
}
if ((Test-Path $lock) -and ((Get-Item $lock).LastWriteTime -gt (Get-Date).AddHours(-2))) {
  Write-Log 'another run is in progress; skipped'
  if ($Show) { Write-Host '另一个复盘正在运行（计划任务或另一个窗口），等它结束后再打开 reports\latest.html。' }
  exit 0
}
Set-Content -Path $lock -Value (Get-Date -Format s)
try {
  $cfg = Get-Content -Raw -Encoding UTF8 (Join-Path $root 'config.json') | ConvertFrom-Json
  $python = $cfg.python
  if (-not (Test-Path $python)) { $python = (Get-Command python -ErrorAction SilentlyContinue).Source }
  $env:PYTHONIOENCODING = 'utf-8'
  Remove-Item Env:STOCK_ROUNDTABLE_ROOT -ErrorAction SilentlyContinue
  $argList = @('-X', 'utf8', (Join-Path $root 'pipeline\run.py'))
  if ($Date) { $argList += @('--date', $Date) }
  if ($Full) { $argList += '--full' }
  Write-Log ('start ' + ($argList -join ' ') + $(if ($Show) { ' (by hand)' } else { '' }))
  if ($Show) {
    Write-Host '最优组合复盘：交易日 15:10 以后才复盘，全部步骤约 12 分钟；周末、节假日和 15:10 以前不会复盘。'
    & $python @argList
    $code = $LASTEXITCODE
  } else {
    $p = Start-Process -FilePath $python -ArgumentList $argList -WorkingDirectory $root -NoNewWindow -Wait -PassThru `
      -RedirectStandardOutput (Join-Path $logs 'last-run.out') -RedirectStandardError (Join-Path $logs 'last-run.err')
    $code = $p.ExitCode
  }
  Write-Log ('exit ' + $code)
  $latest = Join-Path $root 'reports\latest.html'
  if ($Show) {
    $page = if ($Date) { Join-Path $root "reports\$Date.html" } else { $latest }
    if (Test-Path $page) { Write-Host "打开 $page"; Start-Process $page } else { Write-Host '还没有复盘页。' }
  } else {
    $mark = Join-Path $logs 'last-run.json'
    if ($cfg.open_report_when_done -and -not $Date -and (Test-Path $mark)) {
      $m = Get-Content -Raw -Encoding UTF8 $mark | ConvertFrom-Json
      if ($m.new_report -and (Test-Path $latest)) { Start-Process $latest }
    }
  }
  exit $code
} finally {
  Remove-Item $lock -ErrorAction SilentlyContinue
}
