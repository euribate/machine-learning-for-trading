# Workflow Guide — Forked Repository

This repository is a fork of [stefan-jansen/machine-learning-for-trading](https://github.com/stefan-jansen/machine-learning-for-trading).

## Repository Setup

| Remote     | URL                                                              | Purpose               |
|------------|------------------------------------------------------------------|-----------------------|
| `origin`   | https://github.com/euribate/machine-learning-for-trading.git     | Your fork (push/pull) |
| `upstream` | https://github.com/stefan-jansen/machine-learning-for-trading.git | Original repo (pull only) |

## Syncing Upstream Changes

When stefan-jansen publishes updates to the original repository, follow these steps to bring them into your fork.

### Step-by-step

```bash
# 1. Download the latest commits from stefan's repo (does not change your files yet)
git fetch upstream

# 2. Switch to your main branch
git checkout main

# 3. Merge stefan's changes into your local main
git merge upstream/main

# 4. Push the updated main to your fork on GitHub
git push origin main
```

### What each command does

| Command                  | Effect                                                                 |
|--------------------------|------------------------------------------------------------------------|
| `git fetch upstream`     | Downloads new commits from the original repo without modifying files   |
| `git checkout main`      | Switches to the main branch (skip if already on it)                    |
| `git merge upstream/main`| Applies the new upstream changes to your local main branch             |
| `git push origin main`   | Uploads the updated main to your GitHub fork                           |

## Managing Your Own Modifications

Keep personal changes on a dedicated branch so that `main` always mirrors the upstream repo cleanly.

### Creating a branch for your work

```bash
git checkout -b my-experiments
# make your changes...
git add .
git commit -m "describe your changes"
git push origin my-experiments
```

### Updating your branch after an upstream sync

After syncing `main` with upstream (see above), incorporate the new changes into your branch:

```bash
git checkout my-experiments
git merge main
git push origin my-experiments
```

## Handling Merge Conflicts

Conflicts occur when you and stefan modify the same lines in the same file. Git will mark the conflicting sections like this:

```
<<<<<<< HEAD
your version of the code
=======
stefan's version of the code
>>>>>>> upstream/main
```

To resolve:

1. Open the file and decide which version to keep (or combine both).
2. Remove the conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`).
3. Stage and commit the resolved file:

```bash
git add <resolved-file>
git commit -m "resolve merge conflict in <file>"
```

**Tip:** If you never modify files on `main` and only work on your own branches, the upstream sync on `main` will always merge without conflicts.

## Quick Reference

| Task                              | Commands                                                        |
|-----------------------------------|-----------------------------------------------------------------|
| Sync upstream into main           | `git fetch upstream && git checkout main && git merge upstream/main && git push origin main` |
| Create a working branch           | `git checkout -b my-experiments`                                |
| Update your branch after sync     | `git checkout my-experiments && git merge main`                 |
| Push your branch to GitHub        | `git push origin my-experiments`                                |