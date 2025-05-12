#!/usr/bin/env bash
# clean_and_import.sh
# —————————————————————————————————————————
# 1) Clean requirements.txt → clean-requirements.txt
# 2) Backup/init pyproject.toml & set requires-python
# 3) Ensure a dependencies section exists
# 4) Loop: dry-run → patch transitive constraints → retry
# 5) Final poetry add (installs into venv)
# 6) Auto-bump yanked versions:
#      • detect yanked warnings
#      • query PyPI for latest version via python -c
#      • remove old pin & add pkg==<latest>
# 7) Cleanup backups & temps, original requirements.txt untouched
# —————————————————————————————————————————

set -euo pipefail

REQ_FILE="requirements.txt"
EXTRAS_FILE="extras.txt"
CLEAN_FILE="clean-requirements.txt"
BACKUP_TOML="pyproject.toml.bak.$$"
NEW_PROJECT=false
MAX_ROUNDS=10

cleanup(){
  if [[ "$NEW_PROJECT" == true ]]; then
    rm -f pyproject.toml
  elif [[ -f "$BACKUP_TOML" ]]; then
    mv -f "$BACKUP_TOML" pyproject.toml
  fi
  rm -f "$EXTRAS_FILE" "$CLEAN_FILE" install_output.log
}
trap cleanup ERR EXIT

echo "[STEP 1] Backup or initialize pyproject.toml"
if [[ -f pyproject.toml ]]; then
  cp pyproject.toml "$BACKUP_TOML"
else
  poetry init --no-interaction
  NEW_PROJECT=true
fi

echo "[STEP 2] Enforce requires-python"
read MAJ MIN < <(python - <<<'import sys; print(sys.version_info.major, sys.version_info.minor)')
LOW="${MAJ}.${MIN}"
HIGH="${MAJ}.$((MIN+1))"
REQ_PY=">=${LOW},<${HIGH}"
if grep -q '^requires-python' pyproject.toml; then
  sed -i "s|^requires-python.*|requires-python = \"$REQ_PY\"|" pyproject.toml
else
  sed -i "1i requires-python = \"$REQ_PY\"" pyproject.toml
fi

echo "[STEP 3] Ensure dependencies section"
if ! grep -q '^\[tool\.poetry\.dependencies\]' pyproject.toml && \
   ! grep -q '^\[project\.dependencies\]'     pyproject.toml; then
  {
    echo ""
    echo "[tool.poetry.dependencies]"
    echo "python = \"${REQ_PY}\""
  } >> pyproject.toml
fi

echo "[STEP 4] Generate clean-requirements.txt"
grep -E '^\s*(-e |--find-links|--extra-index-url)' "$REQ_FILE" > "$EXTRAS_FILE" || true
grep -v -f "$EXTRAS_FILE" "$REQ_FILE" | sed '/^\s*$/d' | sort -u > "$CLEAN_FILE"
mapfile -t PKGS < "$CLEAN_FILE"
echo "[INFO] To install: ${PKGS[*]}"

echo "[STEP 5] Iterative dry-run + auto-patch"
pattern='depends on ([^[:space:]]+)[[:space:]]\(([^)<>]*[<>][^)]+)\)'

for ((round=1; round<=MAX_ROUNDS; round++)); do
  echo "  • Dry-run attempt #$round…"
  if ERR=$(poetry add --dry-run "${PKGS[@]}" 2>&1); then
    echo "    ✔ Dry-run passed."
    break
  fi

  echo "    ✘ Dry-run failed, parsing conflicts…"
  echo "$ERR"
  patched=false

  while IFS= read -r line; do
    if [[ $line =~ $pattern ]]; then
      pkg="${BASH_REMATCH[1]}"
      cons="${BASH_REMATCH[2]}"
      echo "      ↪ $pkg requires ($cons)"

      # Patch pyproject.toml under correct block
      if grep -q '^\[tool\.poetry\.dependencies\]' pyproject.toml; then
        sed -i "/^\s*${pkg}\s*=/d" pyproject.toml
        sed -i "/^\[tool.poetry.dependencies\]/a ${pkg} = \"${cons}\"" pyproject.toml
      else
        sed -i "/^\[project\.dependencies\]/a ${pkg} = \"${cons}\"" pyproject.toml
      fi
      echo "      → Patched toml: ${pkg} = \"$cons\""

      # Update the install list entry
      for i in "${!PKGS[@]}"; do
        if [[ "${PKGS[$i]}" == "$pkg"* ]]; then
          PKGS[$i]="${pkg}${cons}"
          echo "      → Will install: ${PKGS[$i]}"
        fi
      done

      patched=true
    fi
  done <<< "$ERR"

  $patched || { echo "[ERROR] No more conflicts to patch; adjust toml manually."; exit 1; }
done

echo "[STEP 6] Installing into venv via Poetry"
poetry add "${PKGS[@]}" 2>&1 | tee install_output.log

echo "[STEP 7] Auto-bump yanked versions to latest non-yanked"
mapfile -t YANKED < <(
  grep -Po '^Warning: The locked version [\d\.]+ for \K[^\s]+(?= is a yanked version)' install_output.log || true
)

if (( ${#YANKED[@]} )); then
  echo "[INFO] Found yanked packages: ${YANKED[*]}"
  for pkg in "${YANKED[@]}"; do
    echo "  • Removing old pin for $pkg"
    # note: remove has no --dry-run option
    poetry remove "$pkg" --no-interaction

    echo "  • Querying PyPI for latest $pkg version"
    latest=$(python -c \
      "import json,urllib.request; \
       d=json.load(urllib.request.urlopen('https://pypi.org/pypi/${pkg}/json')); \
       print(d['info']['version'])"
    )
    echo "    ↪ Latest on PyPI is $latest"

    echo "  • Adding $pkg==$latest to pyproject.toml & lockfile"
    poetry add "${pkg}==${latest}" --no-interaction
  done
else
  echo "[INFO] No yanked-version warnings"
fi

echo "[STEP 8] Cleanup backups & temp files"
if [[ "$NEW_PROJECT" == false ]]; then
  rm -f "$BACKUP_TOML"
fi
rm -f "$EXTRAS_FILE" "$CLEAN_FILE" install_output.log
trap - ERR EXIT

echo "[DONE] ✅ Dependencies installed and yanked versions auto-bumped."
