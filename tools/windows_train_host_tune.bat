@echo off
:: Double-click -> UAC -> apply power / update / NVIDIA host tune. No paste needed.
cd /d D:\work\isaac_factory
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\work\isaac_factory\tools\windows_train_host_tune.ps1"
echo.
pause
