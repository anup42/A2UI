param([string]$Ndk = 'C:\Users\anupk\AppData\Local\Android\Sdk\ndk\28.2.13676358')
$ErrorActionPreference = 'Stop'
$libraryRoot = Split-Path -Parent $PSScriptRoot
$compiler = Join-Path $Ndk 'toolchains/llvm/prebuilt/windows-x86_64/bin/clang++.exe'
$source = Join-Path $libraryRoot 'genuicraft/src/main/cpp/fp16_correction.cc'
$destination = Join-Path $libraryRoot 'genuicraft/src/main/jniLibs/arm64-v8a/libGenUiFp16Correction.so'
& $compiler --target=aarch64-linux-android26 -std=c++17 -O2 -Wall -Wextra -Werror -shared -fPIC -static-libstdc++ $source -ldl -llog '-Wl,-z,max-page-size=16384' -o $destination
if ($LASTEXITCODE -ne 0) { throw 'FP16 correction native build failed.' }
Get-FileHash -LiteralPath $destination -Algorithm SHA256
