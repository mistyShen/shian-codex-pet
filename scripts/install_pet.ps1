param(
  [switch]$Build
)

$Root = Split-Path -Parent $PSScriptRoot
$Args = @()
if ($Build) {
  $Args += "--build"
}

if (Get-Command py -ErrorAction SilentlyContinue) {
  py -3 "$Root\scripts\install_pet.py" @Args
} else {
  python "$Root\scripts\install_pet.py" @Args
}
