#!/bin/sh
# Fetch mlsys2025.sty from the official MLSys author kit and check its hashes.
#
# The style file carries no licence statement, so it is not committed. This script
# downloads the kit that the MLSys 2026 call for papers links, verifies the archive and
# the extracted file against pinned SHA-256 digests, and writes the file next to this
# script (gitignored). paper/latexmkrc runs it when the file is missing. Provenance and
# licences: README.md in this directory.
set -eu

KIT_URL="https://media.mlsys.org/Conferences/MLSYS2025/mlsys2025style.zip"
KIT_SHA256="04e77090038f78c985154c71f7a57b7fbb2553ddd3e4e62f431ed420ba3768ef"
STY_SHA256="05a9842992b7ef71851fd2380a1058f83b0faafc106602cabc4c169d372ad8e2"

here=$(cd "$(dirname "$0")" && pwd)
target="$here/mlsys2025.sty"

fail() {
    echo "fetch_mlsys_kit: ERROR: $1" >&2
    echo "fetch_mlsys_kit: the paper cannot be built without template/mlsys2025.sty." >&2
    echo "fetch_mlsys_kit: see paper/template/README.md for the expected file and digests." >&2
    exit 1
}

digest() {
    sha256sum "$1" | cut -d' ' -f1
}

if [ -f "$target" ]; then
    if [ "$(digest "$target")" = "$STY_SHA256" ]; then
        exit 0
    fi
    fail "existing $target has SHA-256 $(digest "$target"), expected $STY_SHA256; delete it to refetch"
fi

command -v curl >/dev/null 2>&1 || fail "curl is not installed"
command -v unzip >/dev/null 2>&1 || fail "unzip is not installed"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fsSL --retry 2 -o "$tmp/kit.zip" "$KIT_URL" \
    || fail "could not download $KIT_URL (offline, or the kit has moved)"
got=$(digest "$tmp/kit.zip")
[ "$got" = "$KIT_SHA256" ] \
    || fail "the kit at $KIT_URL has SHA-256 $got, expected $KIT_SHA256 (the kit changed upstream)"
unzip -q -o "$tmp/kit.zip" "mlsys2025style/mlsys2025.sty" -d "$tmp" \
    || fail "the kit does not contain mlsys2025style/mlsys2025.sty"
got=$(digest "$tmp/mlsys2025style/mlsys2025.sty")
[ "$got" = "$STY_SHA256" ] \
    || fail "mlsys2025.sty has SHA-256 $got, expected $STY_SHA256"
cp "$tmp/mlsys2025style/mlsys2025.sty" "$target"
echo "fetch_mlsys_kit: wrote $target (SHA-256 verified)"
