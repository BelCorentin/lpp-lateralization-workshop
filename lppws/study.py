# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Le Petit Prince *average subject* (English) as a neuralset ``Study``.

This module is a small, self-contained add-on to the public ``neuralfetch``
package. It subclasses :class:`neuralfetch.studies.li2022petit.Li2022Petit`
and swaps the 112 per-subject BOLD files of OpenNeuro ``ds003643`` for the
single *average participant* published by C. Pallier at
https://github.com/chrplr/lpp_average_subject_en — the preprocessed BOLD of
the 49 English speakers, averaged voxel-wise, one 4D NIfTI per run.

It is kept here (rather than inside ``neuralfetch``) so the workshop needs
nothing but ``pip install neuralset neuralfetch`` plus this repository.
"""

from __future__ import annotations

import shutil
import subprocess
import typing as tp
from pathlib import Path

import pandas as pd
from neuralset.events import study
from neuralset.events.etypes import Event
from neuralfetch.studies.li2022petit import Li2022Petit

GITHUB_URL = "https://github.com/chrplr/lpp_average_subject_en"
REPO_NAME = "lpp_average_subject_en"
BRANCH = "main"
_API = f"https://api.github.com/repos/chrplr/{REPO_NAME}/git/trees/{BRANCH}?recursive=1"
_RAW = f"https://raw.githubusercontent.com/chrplr/{REPO_NAME}/{BRANCH}/"

#: Only what the analysis reads. Deliberately excludes ``stimuli/`` (467 MB of
#: *French* audio, useless for the English fMRI) and the tutorial notebook.
WANTED_PREFIXES = ("bold/", "annotation/", "roi_masks/", "mask_lpp_en.nii.gz")


def download_data(dest: str | Path, *, prefixes: tp.Sequence[str] = WANTED_PREFIXES) -> Path:
    """Fetch the average-subject data into ``dest/lpp_average_subject_en``.

    Downloads the individual files over HTTPS (no ``git``, no repository
    history, ~700 MB) and skips anything already present, so it is safe to
    re-run after an interrupted download. Returns the data root.
    """
    import json
    import urllib.request

    root = Path(dest).expanduser().resolve() / REPO_NAME
    root.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(_API, timeout=60) as resp:
        tree = json.load(resp)["tree"]
    files = [
        e["path"]
        for e in tree
        if e["type"] == "blob" and e["path"].startswith(tuple(prefixes))
    ]
    if not files:
        raise RuntimeError(f"Nothing matched {prefixes!r} in {GITHUB_URL}")

    todo = [f for f in files if not (root / f).exists()]
    print(f"{len(files)} files, {len(todo)} to download")
    for i, rel in enumerate(todo, 1):
        out = root / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".part")
        print(f"  [{i}/{len(todo)}] {rel}", flush=True)
        urllib.request.urlretrieve(_RAW + rel, tmp)
        tmp.rename(out)  # rename last, so an interrupted file is never half-used
    return root


def clone_data(dest: str | Path, *, sparse: bool = True) -> Path:
    """``git``-based alternative to :func:`download_data` (keeps a ``.git``).

    Roughly three times the disk of :func:`download_data`, because git also
    stores every file a second time inside ``.git``. Useful only if you want to
    ``git pull`` updates.
    """
    dest = Path(dest).expanduser().resolve()
    repo = dest / REPO_NAME
    if (repo / ".git").exists():
        subprocess.run(["git", "pull", "--ff-only"], cwd=repo, check=True)
        return repo
    if shutil.which("git") is None:
        raise RuntimeError("`git` not found on PATH; use download_data() instead.")
    dest.mkdir(parents=True, exist_ok=True)
    clone = ["git", "clone", "--depth", "1"]
    if sparse:
        clone += ["--filter=blob:none", "--no-checkout"]
    subprocess.run(clone + [GITHUB_URL, str(repo)], check=True)
    if sparse:
        subprocess.run(
            ["git", "sparse-checkout", "set", "--no-cone", *WANTED_PREFIXES],
            cwd=repo,
            check=True,
        )
        subprocess.run(["git", "checkout"], cwd=repo, check=True)
    return repo


class Li2022PetitAverage(Li2022Petit):
    """Average-subject Le Petit Prince fMRI corpus (English).

    One pseudo-subject (``sub-avgEN``), 9 runs = 9 timelines. Each timeline
    yields ``Word`` events (from the TextGrid annotation) and a 4D ``Fmri``
    event, plus an ``Audio`` event when a stimulus wav is present.

    ``path`` may point either directly at a clone of the repository or at a
    parent folder containing ``lpp_average_subject_en/``.

    Expected layout under the data root::

        bold/lpp_en_avg*_run{1..9}_bold.nii.gz   # one averaged 4D NIfTI per run
        annotation/lppEN_section{1..9}.TextGrid  # word timing
        mask_lpp_en.nii.gz                       # brain mask
        roi_masks/                               # language-network ROIs
    """

    #: The repository ships only the *French* wav of section 1, which is of no
    #: use for the English fMRI; the Audio event is therefore opt-in (it also
    #: pulls in `soundfile`).
    load_audio: bool = False

    requirements: tp.ClassVar[tuple[str, ...]] = ("praatio",)
    description: tp.ClassVar[str] = (
        "Le Petit Prince fMRI average subject (English): BOLD averaged across "
        "the 49 English speakers. 9 runs, Word + Fmri (+ Audio) events."
    )

    # neuralset validates the timeline count against this; Li2022Petit declares
    # 1008 (112 subjects x 9 runs), the average subject has 9.
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=9,
        num_subjects=1,
        # first timeline = sub-avgEN run 1: 1517 Word + 1 Fmri event
        num_events_in_query=1518,
        event_types_in_query={"Fmri", "Word"},
        data_shape=(37, 46, 38, 282),  # 4D BOLD for EN run 1
        frequency=0.5,
        fmri_spaces=("custom",),
    )

    def _download(self) -> None:
        download_data(self.path)

    def _data_root(self) -> Path:
        """Directory holding ``bold/`` + ``annotation/``."""
        # neuralset may append the class name to the user-provided path, so the
        # clone can sit at, above, or below `self.path`.
        bases = [self.path, *self.path.parents[:2]]
        for base in bases:
            for cand in (base, base / REPO_NAME, base / "download" / REPO_NAME, base / "download"):
                if (cand / "bold").is_dir():
                    return cand
        return self.path

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        if not self.path.exists():
            raise ValueError(f"No folder {self.path}")
        for run in range(1, 10):
            yield dict(subject="sub-avgEN", lang="EN", run=run)

    # -- events ----------------------------------------------------------

    def _textgrid_path(self, timeline: dict[str, tp.Any]) -> Path:
        tl = timeline
        return (
            self._data_root() / "annotation" / f"lpp{tl['lang']}_section{tl['run']}.TextGrid"
        )

    def _word_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        """Parse the TextGrid annotation into ``Word`` events."""
        txt_grid = self._textgrid_path(timeline)
        if not txt_grid.exists():
            raise FileNotFoundError(
                f"Missing TextGrid for {type(self).__name__}: {txt_grid}. "
                "Run `lppws.study.download_data(...)` (or `study.download()`) first."
            )
        from praatio import textgrid as ptg

        tg = ptg.openTextgrid(str(txt_grid), includeEmptyIntervals=False)
        keys = tg.tierNames
        if len(keys) > 1:
            raise RuntimeError(f"Only one key should be in textgrid, got {keys}")
        # fixes to match text and annotations
        repl = {
            "three_hundred_twenty-five": "325",
            "six_hundred_twelve": "612",
            "one_thousand_nine_hundred_nine": "1909",
            "one_thousand_nine_hundred_twenty": "1920",
            "minster": "minister",
            'na\\i""ve': "naive",
            "coeur": "cœur",
            "oeil": "œil",
            "ll": "il",
            "Â": "",  # garbled encoding artifact in annotation
        }
        # duplicated words happen a lot (missing new-sentence character), merge them
        duplicated = (
            "it the this that i i. they we he she you now and but so there "
            "five twenty phew good".split()
        )
        wordseq: list[dict[str, tp.Any]] = []
        for interval in tg.getTier(keys[0]).entries:
            if interval.label.strip() in ("", "#", "sil"):
                continue
            if interval.label in duplicated and wordseq:
                if wordseq[-1]["text"] == interval.label:
                    wordseq[-1]["duration"] = interval.end - wordseq[-1]["start"]
                    continue
            text = (
                repl.get(interval.label, interval.label)
                .replace("`", "'")
                .replace("«", "")
                .replace("»", "")
            )
            if not text:
                continue
            wordseq.append(
                dict(text=text, start=interval.start, duration=interval.end - interval.start)
            )
        events = pd.DataFrame(wordseq)
        events["type"] = "Word"
        return events

    def _fmri_event(self, timeline: dict[str, tp.Any]) -> dict[str, tp.Any]:
        tl = timeline
        bold_dir = self._data_root() / "bold"
        bold = sorted(bold_dir.glob(f"lpp_{tl['lang'].lower()}_avg*_run{tl['run']}_bold.nii.gz"))
        if not bold:
            raise FileNotFoundError(
                f"No averaged BOLD found for run {tl['run']} in {bold_dir}"
            )
        return dict(
            type="Fmri", start=0, filepath=str(bold[0]), frequency=1.0 / self.TR_FMRI_S
        )

    def _audio_event(self, timeline: dict[str, tp.Any]) -> dict[str, tp.Any]:
        tl = timeline
        stim_dir = self._data_root() / "stimuli"
        # Tolerant to language tag (FR/EN) and section separator ("-"/"_"): the
        # repo currently ships only the French audio of section 1.
        matches = sorted(stim_dir.glob(f"task-lpp*_section*{tl['run']}.wav"))
        if not matches:
            raise FileNotFoundError(f"No stimulus wav for run {tl['run']} in {stim_dir}")
        event = dict(type="Audio", start=0, filepath=str(matches[0]), timeline="tmp")
        return Event.from_dict(event).to_dict()  # populates duration

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        tl = timeline
        events = self._word_events(tl)
        events["language"] = {"en": "english", "fr": "french", "cn": "chinese"}[
            tl["lang"].lower()
        ]
        extra = [self._fmri_event(tl)]
        if self.load_audio:
            try:  # only if the stimulus wavs were downloaded
                extra.append(self._audio_event(tl))
            except FileNotFoundError:
                pass
        out = pd.concat([events, pd.DataFrame(extra)], ignore_index=True)
        out.loc[out.type.isin(["Word", "Sentence", "Text"]), "modality"] = "heard"
        return out.reset_index(drop=True)
