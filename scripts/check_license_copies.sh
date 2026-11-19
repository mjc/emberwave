#!/usr/bin/env bash
# The crates on crates.io carry LICENSE and THIRD_PARTY_NOTICES.md, since a
# package holds nothing outside its own directory (decisions D74 and D75). This
# checks that every copy is identical to the file at the root of the repository,
# so that a change to the root's goes into the copies too.
#
# Usage: scripts/check_license_copies.sh
set -euo pipefail

cd "$(dirname "$0")/.."
status=0
for crate in emberwave-core emberwave-cli; do
    for file in LICENSE THIRD_PARTY_NOTICES.md; do
        if ! cmp -s "$file" "crates/$crate/$file"; then
            echo "crates/$crate/$file differs from $file: cp $file crates/$crate/$file" >&2
            status=1
        fi
    done
done
exit $status
