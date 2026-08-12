[CmdletBinding()]
param(
    [int]$Port = 8000,
    [switch]$Reload
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

# Windows 上 AsyncPostgresSaver 不能运行在 ProactorEventLoop；通过项目内的
# Uvicorn loop factory 强制使用 SelectorEventLoop。端口显式转成字符串，避免
# PowerShell 在传递参数数组时把整数解析成表达式的一部分。
$arguments = @("run", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", [string]$Port, "--loop", "app.uvicorn_loop:selector_loop_factory")
if ($Reload) {
    $arguments += "--reload"
}

Write-Host "Starting RAG Agent on http://127.0.0.1:$Port"
$uvPath = (Get-Command uv.exe -ErrorAction Stop | Select-Object -First 1 -ExpandProperty Source)
& $uvPath @arguments
