# p_1 — git cheat-sheet

Run these in `/content/p_1` (Colab) or in any clone of the repo.

## The 5-command daily ritual

```bash
git pull                      # 1. start of session: get latest
# ...work (edit files, run notebooks)...
git status                    # 2. see what changed
git add -A                    # 3. stage everything
git commit -m "what I did"    # 4. snapshot with a message
git push                      # 5. ship it (end of every session)
```

## Rescue one-liners

| situation                        | command                              |
|----------------------------------|--------------------------------------|
| undo last commit, keep changes   | `git reset --soft HEAD~1`            |
| see what changed                 | `git diff`                           |
| oops, wrong file staged          | `git restore --staged <file>`        |
| session died, check state        | `git status && git log --oneline -3` |
