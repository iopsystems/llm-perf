---
name: release
description: Create a release PR with version bump and changelog update
---

Create a release PR that bumps the version and updates the changelog. After the PR is merged, a GitHub workflow will automatically tag the release and bump to the next development version.

## Arguments

The skill accepts a version level argument:
- `patch` - 0.1.0 -> 0.1.1
- `minor` - 0.1.0 -> 0.2.0
- `major` - 0.1.0 -> 1.0.0
- Or an explicit version like `0.2.0`

Example: `/release minor`

## Steps

1. **Verify prerequisites**:
   - Must be on `main` branch
   - Working directory must be clean
   - Must be up to date with origin/main

   ```bash
   git fetch origin
   if [ "$(git branch --show-current)" != "main" ]; then
     echo "Error: Must be on main branch"
     exit 1
   fi
   if [ -n "$(git status --porcelain)" ]; then
     echo "Error: Working directory not clean"
     exit 1
   fi
   if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]; then
     echo "Error: Not up to date with origin/main"
     exit 1
   fi
   ```

2. **Run local checks**:
   ```bash
   cargo clippy --all-targets --all-features -- -D warnings
   cargo test --lib
   ```
   If checks fail, stop and report the errors.

3. **Determine the new version**:
   ```bash
   # Get current version from Cargo.toml
   CURRENT=$(grep '^version = ' Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
   echo "Current version: $CURRENT"

   # Use cargo-release to calculate new version
   cargo release version <LEVEL> --dry-run 2>&1 | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+'
   ```

4. **Create release branch**:
   ```bash
   NEW_VERSION="X.Y.Z"  # from step 3
   git checkout -b release/v${NEW_VERSION}
   ```

5. **Bump versions using cargo-release**:
   ```bash
   cargo release version <LEVEL> --execute --no-confirm
   ```

6. **Update CHANGELOG.md**:
   - Move items from "Unreleased" section to new version section
   - Add release date
   - Create new empty "Unreleased" section

   The changelog should follow Keep a Changelog format. Ask the user if they want to review/edit the changelog before proceeding.

7. **Commit changes**:

   **CRITICAL**: The commit message MUST start with `release: v` (no other words before the version).
   The `tag-release.yml` workflow matches `startsWith(message, 'release: v')` on the merge commit.
   When GitHub squash-merges a single-commit PR, the commit message becomes the merge commit message.

   ```bash
   git add -A
   git commit -m "release: v${NEW_VERSION}"
   ```

8. **Push and create PR**:
   ```bash
   git push -u origin release/v${NEW_VERSION}

   gh pr create \
     --title "release: v${NEW_VERSION}" \
     --body "$(cat <<EOF
   ## Release v${NEW_VERSION}

   This PR prepares the release of v${NEW_VERSION}.

   ### Changes
   - Version bump
   - Changelog update

   ### After Merge
   The release workflow will automatically:
   1. Create git tag \`v${NEW_VERSION}\`
   2. Build and publish release artifacts (deb/rpm for amd64/arm64)
   3. Bump to next development version (\`-alpha.0\`)

   ---
   See CHANGELOG.md for details.
   EOF
   )"
   ```

9. **Report the PR URL** to the user.

## After PR Merge

When the PR is merged to main, the `tag-release.yml` workflow will:
1. Detect the version from Cargo.toml
2. Create and push the git tag `vX.Y.Z`
3. The `release.yml` workflow then builds deb/rpm packages and creates a GitHub Release
4. Open and auto-merge a PR bumping to next dev version (e.g., `0.2.1-alpha.0`)

## Troubleshooting

- **cargo-release not installed**: `cargo install cargo-release`
- **gh CLI not installed**: `brew install gh` or see https://cli.github.com/
- **Not authenticated with gh**: `gh auth login`
