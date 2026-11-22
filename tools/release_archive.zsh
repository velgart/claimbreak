#!/bin/zsh
# release_archive.zsh -- the archive to deposit on Zenodo for a tagged release, checked by
# installing it into a fresh virtual environment and running the test suite there.
#   zsh tools/release_archive.zsh v1.7.103 ~/claimbreak_work/releases
# Writes OUT/claimbreak-1.7.103.tar.gz (git archive of the tag: no history, without the folders .gitattributes marks
# export-ignore) and OUT/claimbreak-1.7.103.tar.gz.sha256. Refuses to overwrite, to run on a dirty tree, or on a tag
# that is not the version in pyproject.toml. Nothing is uploaded anywhere.
setopt no_bang_hist
TAG=${1:?usage: zsh tools/release_archive.zsh vX.Y.Z OUT_DIR}; OUT=${2:?usage: zsh tools/release_archive.zsh vX.Y.Z OUT_DIR}
V=${TAG#v}; NAME=claimbreak-$V; A=$OUT/$NAME.tar.gz
cd ${0:A:h}/..
[[ -z $(git status --porcelain --untracked-files=no) ]] || { print "STOP: working tree not clean"; exit 1 }
git rev-parse -q --verify "refs/tags/$TAG" > /dev/null || { print "STOP: no tag $TAG"; exit 1 }
grep -q "^version = \"$V\"" <(git show $TAG:pyproject.toml) || { print "STOP: pyproject.toml at $TAG is not version $V"; exit 1 }
[[ -e $A ]] && { print "STOP: $A exists"; exit 1 }
mkdir -p $OUT
git archive --format=tar.gz --prefix=$NAME/ -o $A $TAG
shasum -a 256 $A | tee $A.sha256
print "contents: $(tar tzf $A | wc -l | tr -d ' ') entries; left out: $(git check-attr export-ignore -- docs/* | grep -c ': set$') folder(s)"
# check: a clean install from the archive passes the tests (the fresh environment's own python and claimbreak).
# Editable, as the project is installed everywhere.
T=$(mktemp -d); tar xzf $A -C $T
EXTRAS="[test,xenium]"
python3 -m venv $T/venv || { print "STOP: venv"; exit 1 }
$T/venv/bin/python -m pip install -q -e "$T/$NAME$EXTRAS" || { print "STOP: pip install from the archive failed"; exit 1 }
(cd $T/$NAME && $T/venv/bin/python -m pytest -q -p no:cacheprovider > $T/pytest.log 2>&1); rc=$?
tail -2 $T/pytest.log
(( rc == 0 )) || { print "STOP: tests failed in the clean install ($T/pytest.log kept)"; exit 1 }
$T/venv/bin/claimbreak kept-ratio --help > /dev/null || { print "STOP: claimbreak kept-ratio not runnable"; exit 1 }
print "clean install: tests pass, claimbreak kept-ratio runs"
rm -rf $T
print "deposit $A on Zenodo (upload type: software); metadata in .zenodo.json and CITATION.cff"
