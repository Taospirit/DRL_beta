# git rev-list --objects --all | grep 4cc1f9dcef1004355d2a595d45808e99f100dc4d
# git log --pretty=oneline --branches --  app/src/assets/img/FS.mp4

# git filter-branch --tree-filter "rm -f {filepath}" -- --all
# git filter-branch --index-filter 'git rm --cached --ignore-unmatch  app/src/assets/img/FS.mp4' -- --all
git filter-branch --force --index-filter 'git rm -rf --cached --ignore-unmatch beta/save/test_sac_mac/RunTime440.jpg' --prune-empty --tag-name-filter cat -- --all


rm -Rf .git/refs/original
rm -Rf .git/logs/
git gc
git prune