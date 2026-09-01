$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

function Invoke-Checked {
  param(
    [Parameter(Mandatory = $true)]
    [string] $Program,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Arguments
  )

  & $Program @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed with exit code $LASTEXITCODE`: $Program $Arguments"
  }
}

Write-Host "Building QuantitativeStrictAT"
Invoke-Checked -Program lake -Arguments @("build", "QuantitativeStrictAT")

Write-Host "Checking Main.lean and the public Main API"
Invoke-Checked -Program lake -Arguments @(
  "env",
  "lean",
  "-o",
  ".lake/build/lib/lean/Main.olean",
  "Main.lean"
)

$apiCheck = Join-Path ([System.IO.Path]::GetTempPath()) `
  ("rsat-api-check-" + [System.Guid]::NewGuid().ToString("N") + ".lean")
$apiSource = @'
import Main

example (beta : Real) {h : Real} (hh : 0 < h) :
    Main.ReplicaSymmetricFixedPointClaim beta h :=
  Main.replicaSymmetricFixedPointClaim_of_pos_field beta hh

example : Main.StrictAT_main :=
  Main.strictAT_main

example (beta h : Real) : Main.OverlapCLTClaim beta h :=
  Main.strictAT_overlapCLT_weak beta h
'@

try {
  [System.IO.File]::WriteAllText(
    $apiCheck,
    $apiSource,
    [System.Text.UTF8Encoding]::new($false)
  )
  Invoke-Checked -Program lake -Arguments @("env", "lean", $apiCheck)
}
finally {
  if (Test-Path -LiteralPath $apiCheck) {
    Remove-Item -LiteralPath $apiCheck -Force
  }
}

Write-Host "Scanning project Lean sources for placeholders and local axioms"
$pattern = '(?<![A-Za-z0-9_])(sorry|admit|sorryAx|axiom)(?![A-Za-z0-9_])'
$sourceFiles = @((Get-Item -LiteralPath Main.lean)) +
  @(Get-ChildItem -LiteralPath Lemmas -Recurse -File -Filter *.lean)
$matches = $sourceFiles | Select-String -Pattern $pattern
if ($matches) {
  $matches | ForEach-Object {
    Write-Error ("{0}:{1}:{2}" -f $_.Path, $_.LineNumber, $_.Line)
  }
  throw "Verification failed: inappropriate placeholders or project-local axiom declarations were found."
}

Write-Host "Verification succeeded: build, public endpoint, API, and source-integrity checks passed."
