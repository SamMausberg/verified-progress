#!/usr/bin/env bash
# Refuse to commit this machine's identity. The repository is public, and on this cloud box the
# hostname is the public IP address written with dashes, so a server id or launch record that
# carries the hostname publishes the address.
#
#   scripts/check_host_identity.sh [file...]
#
# With no arguments it checks the files staged for commit (pre-commit runs it that way, so the
# config's evidence/ exclusion does not hide evidence files from it). It looks for the hostname
# when that is an IP address in dashed form, and for every public IPv4 address `hostname -I`
# reports, dotted and dashed. Matches are reported by file and line, without the value.
set -euo pipefail

patterns=()
host="$(hostname 2>/dev/null || true)"
if [[ $host =~ ^[0-9]{1,3}(-[0-9]{1,3}){3}$ ]]; then
  patterns+=("$host" "${host//-/.}")
fi
for addr in $(hostname -I 2>/dev/null || true); do
  case "$addr" in
    *:* | 127.* | 10.* | 192.168.* | 169.254.*) continue ;;
    172.1[6-9].* | 172.2[0-9].* | 172.3[01].*) continue ;;
  esac
  patterns+=("$addr" "${addr//./-}")
done
[ "${#patterns[@]}" -gt 0 ] || exit 0

staged=""
if [ "$#" -gt 0 ]; then
  files=("$@")
else
  # NUL-delimited, so unusual paths are not quoted; T covers a symlink replaced by a file.
  staged=1
  mapfile -d '' -t files < <(git diff --cached --name-only -z --diff-filter=ACMRT)
fi
args=()
for p in "${patterns[@]}"; do args+=(-e "$p"); done

status=0
for f in "${files[@]}"; do
  # Staged content, not the working copy, when checking a commit.
  if [ -n "$staged" ]; then
    hits="$(git cat-file blob ":$f" 2>/dev/null | grep -n -I -F "${args[@]}" | cut -d: -f1 | tr '\n' ' ' || true)"
  else
    [ -f "$f" ] || continue
    hits="$(grep -n -I -F "${args[@]}" "$f" | cut -d: -f1 | tr '\n' ' ' || true)"
  fi
  if [ -n "$hits" ]; then
    echo "$f: line(s) ${hits% } contain this machine's hostname or IP address" >&2
    status=1
  fi
done
exit "$status"
