#!/usr/bin/env bash
# Refuse to commit this machine's identity. The repository is public, and on this cloud box the
# hostname is the public IP address written with dashes, so a server id or launch record that
# carries the hostname publishes the address.
#
#   scripts/check_host_identity.sh [file...]
#
# With no arguments it checks the files staged for commit (pre-commit runs it that way, so the
# config's evidence/ exclusion does not hide evidence files from it). It looks for the hostname
# when that is a global IP address in dashed form, and for every public IPv4 address `hostname -I`
# reports (dotted and dashed), and for every global IPv6 address it reports (compressed and
# fully expanded spellings, any case; other spellings are not matched). File paths are checked
# as well as contents. Binary files are
# scanned as text too (a PDF or binary artifact can embed the address); a compressed stream
# can still hide it, so check generated archives at the source. Matches are reported by file
# and line, without the value.
set -euo pipefail

patterns=()
host="$(hostname 2>/dev/null || true)"
if [[ $host =~ ^[0-9]{1,3}(-[0-9]{1,3}){3}$ ]]; then
  # The same rule as for the addresses below: only a global address is a pattern.
  if python3 -c 'import ipaddress, sys
sys.exit(0 if ipaddress.ip_address(sys.argv[1]).is_global else 1)' "${host//-/.}" 2>/dev/null ||
    ! command -v python3 >/dev/null; then
    patterns+=("$host" "${host//-/.}")
  fi
fi
for addr in $(hostname -I 2>/dev/null || true); do
  # Global addresses only, classified with Python's ipaddress (private, CGNAT, link-local,
  # unique-local and other reserved ranges are skipped): IPv4 dotted and dashed, IPv6 compressed
  # and fully expanded (the search ignores case). If Python fails, the address is used as given.
  mapfile -t forms < <(python3 -c 'import ipaddress, sys
a = ipaddress.ip_address(sys.argv[1].split("%")[0])
if a.is_global:
    if a.version == 4:
        print(a)
        print(str(a).replace(".", "-"))
    else:
        print(a.compressed)
        print(a.exploded)' "$addr" 2>/dev/null || echo "$addr")
  patterns+=("${forms[@]}")
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
  # The path itself is published too; a path that names the identity is not printed.
  shown="$f"
  for p in "${patterns[@]}"; do
    if [[ ${f,,} == *"${p,,}"* ]]; then
      shown="(a staged path, not shown)"
      echo "$shown: the path contains this machine's hostname or IP address" >&2
      status=1
      break
    fi
  done
  # Staged content, not the working copy, when checking a commit.
  if [ -n "$staged" ]; then
    hits="$(git cat-file blob ":$f" 2>/dev/null | grep -n -a -i -F "${args[@]}" | cut -d: -f1 | tr '\n' ' ' || true)"
  else
    [ -f "$f" ] || continue
    hits="$(grep -n -a -i -F "${args[@]}" -- "$f" | cut -d: -f1 | tr '\n' ' ' || true)"
  fi
  if [ -n "$hits" ]; then
    echo "$shown: line(s) ${hits% } contain this machine's hostname or IP address" >&2
    status=1
  fi
done
exit "$status"
