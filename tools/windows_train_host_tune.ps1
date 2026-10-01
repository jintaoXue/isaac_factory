#Requires -RunAsAdministrator
# One-shot: power + Windows Update anti-reboot + NVIDIA hints
# Right-click PowerShell -> Run as administrator, then:
#   Set-ExecutionPolicy -Scope Process Bypass -Force
#   & D:\work\isaac_factory\tools\windows_train_host_tune.ps1

$ErrorActionPreference = 'Stop'
Write-Host '== Power: Ultimate Performance, no sleep on AC =='
powercfg /setactive e9a42b02-d5df-448d-aa00-03f14749eb61
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change monitor-timeout-ac 0
powercfg /change disk-timeout-ac 0
powercfg /SETACVALUEINDEX SCHEME_CURRENT SUB_PCIEXPRESS ASPM 0
powercfg /SETACTIVE SCHEME_CURRENT
powercfg /hibernate off
powercfg /getactivescheme

Write-Host '== Windows Update: pause ~35d + no auto-reboot when logged on =='
$ux = 'HKLM:\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings'
New-Item -Path $ux -Force | Out-Null
$start = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
$end = (Get-Date).AddDays(35).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
New-ItemProperty -Path $ux -Name 'IsActiveHoursEnabled' -Value 1 -PropertyType DWord -Force | Out-Null
New-ItemProperty -Path $ux -Name 'ActiveHoursStart' -Value 8 -PropertyType DWord -Force | Out-Null
New-ItemProperty -Path $ux -Name 'ActiveHoursEnd' -Value 2 -PropertyType DWord -Force | Out-Null
New-ItemProperty -Path $ux -Name 'PauseUpdatesStartTime' -Value $start -PropertyType String -Force | Out-Null
New-ItemProperty -Path $ux -Name 'PauseUpdatesExpiryTime' -Value $end -PropertyType String -Force | Out-Null
New-ItemProperty -Path $ux -Name 'PauseFeatureUpdatesStartTime' -Value $start -PropertyType String -Force | Out-Null
New-ItemProperty -Path $ux -Name 'PauseFeatureUpdatesEndTime' -Value $end -PropertyType String -Force | Out-Null
New-ItemProperty -Path $ux -Name 'PauseQualityUpdatesStartTime' -Value $start -PropertyType String -Force | Out-Null
New-ItemProperty -Path $ux -Name 'PauseQualityUpdatesEndTime' -Value $end -PropertyType String -Force | Out-Null

$au = 'HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU'
New-Item -Path $au -Force | Out-Null
# 3 = auto download + notify install (no silent reboot storm)
New-ItemProperty -Path $au -Name 'AUOptions' -Value 3 -PropertyType DWord -Force | Out-Null
New-ItemProperty -Path $au -Name 'NoAutoRebootWithLoggedOnUsers' -Value 1 -PropertyType DWord -Force | Out-Null

$wu = 'HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate'
New-Item -Path $wu -Force | Out-Null
New-ItemProperty -Path $wu -Name 'NoAutoRebootWithLoggedOnUsers' -Value 1 -PropertyType DWord -Force | Out-Null

Write-Host '== NVIDIA: Prefer Maximum Performance (Base Profile) via nvidia-smi / note =='
# Lock application clocks near boost when supported (ignored on some WDDM setups)
try {
  nvidia-smi -i 0 -pl 450 2>$null | Out-Host
} catch {}
Write-Host @'

Manual (once): NVIDIA Control Panel
  1. Manage 3D settings -> Global Settings
  2. Power management mode = Prefer maximum performance
  3. Apply

Also: Settings -> Windows Update -> Pause updates (confirm UI matches ~35 days)

Done.
'@
