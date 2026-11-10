# 注册计划任务 Review-OptimalCombo（工作日 15:10 复盘、16:40 补跑或数据核对），并在桌面放“最优组合复盘”（打开最新一期）
# 和“立即复盘”（马上跑一遍）两个快捷方式。
#   powershell -ExecutionPolicy Bypass -File D:\review\install_task.ps1
# 卸载：Unregister-ScheduledTask -TaskName Review-OptimalCombo -Confirm:$false，再删除桌面上的快捷方式。
$root = $PSScriptRoot
$action = New-ScheduledTaskAction -Execute 'powershell.exe' `
  -Argument ('-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "{0}"' -f (Join-Path $root 'run_review.ps1')) `
  -WorkingDirectory $root
$days = 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'
$triggers = @((New-ScheduledTaskTrigger -Weekly -DaysOfWeek $days -At '15:10'),
              (New-ScheduledTaskTrigger -Weekly -DaysOfWeek $days -At '16:40'))
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew `
  -ExecutionTimeLimit (New-TimeSpan -Hours 2) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName 'Review-OptimalCombo' -Action $action -Trigger $triggers -Settings $settings `
  -Description '最优组合每日复盘：D:\review\run_review.ps1（工作日 15:10 复盘，16:40 补跑或数据核对）' -Force | Out-Null
$desktop = [Environment]::GetFolderPath('Desktop')
$shell = New-Object -ComObject WScript.Shell
$lnk = $shell.CreateShortcut((Join-Path $desktop '最优组合复盘.lnk'))
$lnk.TargetPath = Join-Path $root 'reports\latest.html'
$lnk.Description = '最优组合每日复盘（最新一期）'
$lnk.Save()
$run = $shell.CreateShortcut((Join-Path $desktop '立即复盘.lnk'))
$run.TargetPath = Join-Path $root '立即复盘.cmd'
$run.WorkingDirectory = $root
$run.Description = '立即运行最优组合复盘（交易日 15:10 以后），显示进度，完成后打开复盘页'
$run.Save()
Get-ScheduledTask -TaskName 'Review-OptimalCombo' | Get-ScheduledTaskInfo | Select-Object TaskName, NextRunTime
