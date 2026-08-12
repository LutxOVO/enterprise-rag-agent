[CmdletBinding()]
param(
    [switch]$SkipStaticChecks
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($env:TEST_DATABASE_URL)) {
    $databaseUrl = $env:DATABASE_URL
    if ([string]::IsNullOrWhiteSpace($databaseUrl) -and (Test-Path .env)) {
        $databaseLine = Get-Content -Encoding UTF8 .env |
            Where-Object { $_ -match '^DATABASE_URL=' } |
            Select-Object -First 1
        if ($databaseLine) {
            $databaseUrl = ($databaseLine -split '=', 2)[1].Trim().Trim('"')
        }
    }
    if ([string]::IsNullOrWhiteSpace($databaseUrl)) {
        $databaseUrl = "postgresql+psycopg://rag:local-only-change-me@localhost:5432/rag"
    }
    $env:TEST_DATABASE_URL = $databaseUrl -replace '/rag(?=\?|$)', '/rag_test'
}

if ($env:TEST_DATABASE_URL -notmatch "/rag_test(?:\?|$)") {
    throw "TEST_DATABASE_URL must point to the dedicated rag_test database."
}

Write-Host "Running tests against $($env:TEST_DATABASE_URL -replace ':[^:@/]+@', ':***@')"
@'
import os
import psycopg

database_url = os.environ["TEST_DATABASE_URL"].replace("postgresql+psycopg://", "postgresql://")
try:
    with psycopg.connect(database_url, connect_timeout=3) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
except Exception as exc:
    raise SystemExit(f"PostgreSQL preflight failed: {type(exc).__name__}: {exc}") from exc

print("PostgreSQL preflight passed.")
'@ | uv run python -u -
if ($LASTEXITCODE -ne 0) {
    throw "PostgreSQL is unavailable. Start PostgreSQL and rerun scripts/test.ps1."
}

uv run pytest -q
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if (-not $SkipStaticChecks) {
    uv run python -m compileall -q app scripts tests
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    node --check app/static/app.js
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Write-Host "All PostgreSQL-backed tests and static checks passed."
