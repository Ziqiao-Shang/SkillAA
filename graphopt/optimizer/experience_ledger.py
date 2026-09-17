"""Optimizer-side experience ledger (NOT part of agent skill / current_graph).

Records badcase opinions that drive GraphPatch proposals. Persisted separately
(e.g. ``out_root/experience_ledger.json``). Never rendered into the Agent prompt.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _slug(text: str, n: int = 48) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip().lower()).strip("_")
    return (s[:n] or "opinion").strip("_")


@dataclass
class ExperienceNote:
    id: str
    summary: str
    n_seen: int = 1
    n_fail: int = 1
    last_step: int = 0
    status: str = "open"  # open | applied | rejected
    suggested_ops: list[str] = field(default_factory=list)
    related_node_ids: list[str] = field(default_factory=list)
    source_traj_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperienceNote":
        return cls(
            id=str(d.get("id") or ""),
            summary=str(d.get("summary") or ""),
            n_seen=int(d.get("n_seen") or 0),
            n_fail=int(d.get("n_fail") or 0),
            last_step=int(d.get("last_step") or 0),
            status=str(d.get("status") or "open"),
            suggested_ops=[str(x) for x in (d.get("suggested_ops") or [])],
            related_node_ids=[str(x) for x in (d.get("related_node_ids") or [])],
            source_traj_ids=[str(x) for x in (d.get("source_traj_ids") or [])],
        )


@dataclass
class ExperienceLedger:
    notes: dict[str, ExperienceNote] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "ExperienceLedger":
        raw = (d or {}).get("notes") or []
        if isinstance(raw, dict):
            notes = {k: ExperienceNote.from_dict({**v, "id": k}) for k, v in raw.items()}
        else:
            notes = {n["id"]: ExperienceNote.from_dict(n) for n in raw if n.get("id")}
        return cls(notes=notes)

    def to_dict(self) -> dict[str, Any]:
        ordered = sorted(self.notes.values(), key=lambda n: (-n.n_fail, -n.n_seen, n.id))
        return {"notes": [n.to_dict() for n in ordered]}

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ExperienceLedger":
        p = Path(path)
        if not p.is_file():
            return cls()
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def upsert(
        self,
        summary: str,
        *,
        step: int = 0,
        traj_id: str = "",
        suggested_ops: list[str] | None = None,
        related_node_ids: list[str] | None = None,
        is_fail: bool = True,
    ) -> ExperienceNote:
        key = _slug(summary)
        note = self.notes.get(key)
        if note is None:
            note = ExperienceNote(
                id=key,
                summary=summary.strip(),
                n_seen=1,
                n_fail=1 if is_fail else 0,
                last_step=step,
                status="open",
                suggested_ops=list(suggested_ops or []),
                related_node_ids=list(related_node_ids or []),
                source_traj_ids=[traj_id] if traj_id else [],
            )
            self.notes[key] = note
            return note
        note.n_seen += 1
        if is_fail:
            note.n_fail += 1
        note.last_step = step
        if note.status == "rejected":
            note.status = "open"
        if traj_id and traj_id not in note.source_traj_ids:
            note.source_traj_ids.append(traj_id)
            note.source_traj_ids = note.source_traj_ids[-20:]
        for op in suggested_ops or []:
            if op not in note.suggested_ops:
                note.suggested_ops.append(op)
        for nid in related_node_ids or []:
            if nid not in note.related_node_ids:
                note.related_node_ids.append(nid)
        return note

    def mark_applied(self, note_ids: list[str]) -> None:
        for nid in note_ids:
            if nid in self.notes:
                self.notes[nid].status = "applied"

    def mark_rejected(self, note_ids: list[str]) -> None:
        for nid in note_ids:
            if nid in self.notes:
                self.notes[nid].status = "rejected"

    def format_for_optimizer(self, *, max_notes: int = 16) -> str:
        open_notes = [n for n in self.notes.values() if n.status == "open"]
        open_notes.sort(key=lambda n: (-n.n_fail, -n.n_seen, n.id))
        if not open_notes:
            return "(no open experience notes yet)"
        lines = [
            "## Optimizer experience ledger (NOT in Agent prompt)",
            "Use these recurring failure opinions when proposing GraphPatch edits to the rule graph.",
            "Do not copy this ledger into the skill document.",
            "",
        ]
        for n in open_notes[:max_notes]:
            rel = ",".join(n.related_node_ids) if n.related_node_ids else "-"
            lines.append(
                f"- `{n.id}` ×{n.n_fail} fails / {n.n_seen} seen | nodes={rel} | {n.summary}"
            )
        return "\n".join(lines)


# Heuristic opinion extractors for dry-run / bootstrap before optimizer chat.
_PATTERNS: list[tuple[tuple[str, ...], str, list[str]]] = [
    (
        ("cabinet", "drawer", "fridge", "container", "closed"),
        "Open closed containers before concluding the object is missing.",
        ["G02", "G03", "H02"],
    ),
    (
        ("verify", "final state", "confirm", "goal"),
        "Verify environment state after key actions before ending the episode.",
        ["G08", "G07"],
    ),
    (
        ("search", "look", "explore", "not find"),
        "Keep a persistent search ledger and prefer unvisited receptacles.",
        ["G03", "H01", "H02"],
    ),
]


def ingest_rollouts(
    ledger: ExperienceLedger,
    results: list[dict[str, Any]],
    *,
    step: int = 0,
) -> list[str]:
    """Summarize badcases into ledger notes; return touched note ids."""
    touched: list[str] = []
    for r in results:
        hard = r.get("hard")
        if hard and float(hard or 0) >= 1e-9:
            continue
        blob = " ".join(
            [
                str(r.get("fail_reason") or ""),
                str(r.get("task_description") or ""),
                str(r.get("trajectory") or ""),
            ]
        ).lower()
        matched = False
        for keys, summary, nodes in _PATTERNS:
            if any(k in blob for k in keys):
                note = ledger.upsert(
                    summary,
                    step=step,
                    traj_id=str(r.get("id") or ""),
                    related_node_ids=nodes,
                    suggested_ops=["update_node"],
                    is_fail=True,
                )
                touched.append(note.id)
                matched = True
        if not matched:
            reason = (str(r.get("fail_reason") or "unspecified failure")).strip()
            note = ledger.upsert(
                f"Investigate failure: {reason[:160]}",
                step=step,
                traj_id=str(r.get("id") or ""),
                suggested_ops=["update_node"],
                is_fail=True,
            )
            touched.append(note.id)
    # unique preserve order
    return list(dict.fromkeys(touched))
