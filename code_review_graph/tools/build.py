"""Tool 1: build_or_update_graph + run_postprocess."""

from __future__ import annotations

import logging
import sqlite3
import time
from typing import Any

from ..incremental import full_build, incremental_update
from ._common import _get_store

logger = logging.getLogger(__name__)


def _run_postprocess(
    store: Any,
    build_result: dict[str, Any],
    postprocess: str,
    full_rebuild: bool = False,
    changed_files: list[str] | None = None,
) -> list[str]:
    """Run post-build steps based on *postprocess* level.

    When *full_rebuild* is False and *changed_files* are available,
    uses incremental flow/community detection for faster updates.

    Returns a list of warning strings (empty on success).
    """
    warnings: list[str] = []
    build_result["postprocess_level"] = postprocess

    if postprocess == "none":
        return warnings

    # -- Signatures + FTS (fast, always run unless "none") --
    try:
        rows = store.get_nodes_without_signature()
        sig_updates: list[tuple[str, int]] = []
        for row in rows:
            node_id, name, kind, params, ret = (
                row[0], row[1], row[2], row[3], row[4],
            )
            if kind in ("Function", "Test"):
                sig = f"def {name}({params or ''})"
                if ret:
                    sig += f" -> {ret}"
            elif kind == "Class":
                sig = f"class {name}"
            else:
                sig = name
            sig_updates.append((sig[:512], node_id))
        if sig_updates:
            store._conn.executemany(
                "UPDATE nodes SET signature = ? WHERE id = ?", sig_updates,
            )
        store.commit()
        build_result["signatures_updated"] = True
    except (sqlite3.OperationalError, TypeError, KeyError) as e:
        logger.warning("Signature computation failed: %s", e)
        warnings.append(f"Signature computation failed: {type(e).__name__}: {e}")

    try:
        if not full_rebuild and changed_files:
            from code_review_graph.search import update_fts_index

            fts_count = update_fts_index(store, changed_files)
        else:
            from code_review_graph.search import rebuild_fts_index

            fts_count = rebuild_fts_index(store)
        build_result["fts_indexed"] = fts_count
        build_result["fts_rebuilt"] = True
    except (sqlite3.OperationalError, ImportError) as e:
        logger.warning("FTS index rebuild failed: %s", e)
        warnings.append(f"FTS index rebuild failed: {type(e).__name__}: {e}")

    if postprocess == "minimal":
        return warnings

    # -- Expensive: flows + communities (only for "full") --
    use_incremental = not full_rebuild and bool(changed_files)

    try:
        if use_incremental:
            from code_review_graph.flows import incremental_trace_flows

            count = incremental_trace_flows(store, changed_files)
        else:
            from code_review_graph.flows import store_flows as _store_flows
            from code_review_graph.flows import trace_flows as _trace_flows

            flows = _trace_flows(store)
            count = _store_flows(store, flows)
        build_result["flows_detected"] = count
    except (sqlite3.OperationalError, ImportError) as e:
        logger.warning("Flow detection failed: %s", e)
        warnings.append(f"Flow detection failed: {type(e).__name__}: {e}")

    try:
        if use_incremental:
            from code_review_graph.communities import (
                incremental_detect_communities,
            )

            count = incremental_detect_communities(store, changed_files)
        else:
            from code_review_graph.communities import (
                detect_communities as _detect_communities,
            )
            from code_review_graph.communities import (
                store_communities as _store_communities,
            )

            comms = _detect_communities(store)
            count = _store_communities(store, comms)
        build_result["communities_detected"] = count
    except (sqlite3.OperationalError, ImportError) as e:
        logger.warning("Community detection failed: %s", e)
        warnings.append(f"Community detection failed: {type(e).__name__}: {e}")

    # -- Compute pre-computed summary tables --
    try:
        _compute_summaries(store)
        build_result["summaries_computed"] = True
    except (sqlite3.OperationalError, Exception) as e:
        logger.warning("Summary computation failed: %s", e)
        warnings.append(f"Summary computation failed: {type(e).__name__}: {e}")

    store.set_metadata(
        "last_postprocessed_at", time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    store.set_metadata("postprocess_level", postprocess)

    return warnings


def _compute_summaries(store: Any) -> None:
    """Populate community_summaries, flow_snapshots, and risk_index tables."""
    import json as _json

    conn = store._conn

    # -- community_summaries --
    try:
        conn.execute("DELETE FROM community_summaries")
        rows = conn.execute(
            "SELECT id, name, size, dominant_language FROM communities"
        ).fetchall()
        for r in rows:
            cid, cname, csize, clang = r[0], r[1], r[2], r[3]
            # Top 5 symbols by in+out edge count
            top_symbols = conn.execute(
                "SELECT n.name FROM nodes n "
                "LEFT JOIN edges e1 ON e1.source_qualified = n.qualified_name "
                "LEFT JOIN edges e2 ON e2.target_qualified = n.qualified_name "
                "WHERE n.community_id = ? AND n.kind != 'File' "
                "GROUP BY n.id ORDER BY COUNT(e1.id) + COUNT(e2.id) DESC "
                "LIMIT 5",
                (cid,),
            ).fetchall()
            key_syms = _json.dumps([s[0] for s in top_symbols])
            # Auto-generate purpose from common file path prefix
            file_rows = conn.execute(
                "SELECT DISTINCT file_path FROM nodes WHERE community_id = ? LIMIT 20",
                (cid,),
            ).fetchall()
            paths = [fr[0] for fr in file_rows]
            purpose = ""
            if paths:
                from os.path import commonprefix
                prefix = commonprefix(paths)
                if "/" in prefix:
                    purpose = prefix.rsplit("/", 1)[0].split("/")[-1] if "/" in prefix else ""
            conn.execute(
                "INSERT OR REPLACE INTO community_summaries "
                "(community_id, name, purpose, key_symbols, size, dominant_language) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (cid, cname, purpose, key_syms, csize, clang or ""),
            )
    except sqlite3.OperationalError:
        pass  # Table may not exist yet

    # -- flow_snapshots (batch: pre-load node names instead of per-id lookups) --
    try:
        conn.execute("DELETE FROM flow_snapshots")
        rows = conn.execute(
            "SELECT id, name, entry_point_id, criticality, node_count, "
            "file_count, path_json FROM flows"
        ).fetchall()

        # Pre-load all node id -> (name, qualified_name) in one query
        node_lookup: dict[int, tuple[str, str]] = {}
        for nr in conn.execute(
            "SELECT id, name, qualified_name FROM nodes"
        ).fetchall():
            node_lookup[nr[0]] = (nr[1], nr[2])

        snapshot_rows: list[tuple] = []
        for r in rows:
            fid, fname, ep_id = r[0], r[1], r[2]
            crit, ncount, fcount = r[3], r[4], r[5]
            ep_info = node_lookup.get(ep_id)
            ep_name = ep_info[1] if ep_info else str(ep_id)
            path_ids = _json.loads(r[6]) if r[6] else []
            critical_path: list[str] = []
            if path_ids:
                critical_path.append(ep_name)
                if len(path_ids) > 2:
                    for nid in path_ids[1:4]:
                        ni = node_lookup.get(nid)
                        if ni:
                            critical_path.append(ni[0])
                if len(path_ids) > 1:
                    li = node_lookup.get(path_ids[-1])
                    if li and li[0] not in critical_path:
                        critical_path.append(li[0])
            snapshot_rows.append((
                fid, fname, ep_name, _json.dumps(critical_path),
                crit, ncount, fcount,
            ))

        if snapshot_rows:
            conn.executemany(
                "INSERT OR REPLACE INTO flow_snapshots "
                "(flow_id, name, entry_point, critical_path, criticality, "
                "node_count, file_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
                snapshot_rows,
            )
    except sqlite3.OperationalError:
        pass

    # -- risk_index (batch: 2 aggregate queries instead of 2N per-node queries) --
    try:
        conn.execute("DELETE FROM risk_index")
        nodes = conn.execute(
            "SELECT id, qualified_name, name FROM nodes "
            "WHERE kind IN ('Function', 'Class', 'Test')"
        ).fetchall()

        # Pre-compute caller counts in one query
        caller_counts: dict[str, int] = {}
        for r in conn.execute(
            "SELECT target_qualified, COUNT(*) FROM edges "
            "WHERE kind = 'CALLS' GROUP BY target_qualified"
        ).fetchall():
            caller_counts[r[0]] = r[1]

        # Pre-compute test coverage in one query
        tested_set: set[str] = set()
        for r in conn.execute(
            "SELECT DISTINCT source_qualified FROM edges "
            "WHERE kind = 'TESTED_BY'"
        ).fetchall():
            tested_set.add(r[0])

        security_kw = {
            "auth", "login", "password", "token", "session", "crypt",
            "secret", "credential", "permission", "sql", "execute",
        }
        risk_rows: list[tuple] = []
        for n in nodes:
            nid, qn, name = n[0], n[1], n[2]
            cc = caller_counts.get(qn, 0)
            coverage = "tested" if qn in tested_set else "untested"
            name_lower = name.lower()
            sec_relevant = 1 if any(kw in name_lower for kw in security_kw) else 0
            risk = 0.0
            if cc > 10:
                risk += 0.3
            elif cc > 3:
                risk += 0.15
            if coverage == "untested":
                risk += 0.3
            if sec_relevant:
                risk += 0.4
            risk = min(risk, 1.0)
            risk_rows.append((nid, qn, risk, cc, coverage, sec_relevant))

        if risk_rows:
            conn.executemany(
                "INSERT OR REPLACE INTO risk_index "
                "(node_id, qualified_name, risk_score, caller_count, "
                "test_coverage, security_relevant, last_computed) "
                "VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
                risk_rows,
            )
    except sqlite3.OperationalError:
        pass

    conn.commit()


def build_or_update_graph(
    full_rebuild: bool = False,
    repo_root: str | None = None,
    base: str = "HEAD~1",
    postprocess: str = "full",
) -> dict[str, Any]:
    """Build or incrementally update the code knowledge graph.

    Args:
        full_rebuild: If True, re-parse every file. If False (default),
                      only re-parse files changed since ``base``.
        repo_root: Path to the repository root. Auto-detected if omitted.
        base: Git ref for incremental diff (default: HEAD~1).
        postprocess: Post-processing level after build:
            ``"full"`` (default) — signatures, FTS, flows, communities.
            ``"minimal"`` — signatures + FTS only (fast, keeps search working).
            ``"none"`` — skip all post-processing (raw parse only).

    Returns:
        Summary with files_parsed/updated, node/edge counts, and errors.
    """
    store, root = _get_store(repo_root)
    try:
        if full_rebuild:
            result = full_build(root, store)
            build_result = {
                "status": "ok",
                "build_type": "full",
                "summary": (
                    f"Full build complete: parsed {result['files_parsed']} files, "
                    f"created {result['total_nodes']} nodes and "
                    f"{result['total_edges']} edges."
                ),
                **result,
            }
        else:
            result = incremental_update(root, store, base=base)
            if result["files_updated"] == 0:
                return {
                    "status": "ok",
                    "build_type": "incremental",
                    "summary": "No changes detected. Graph is up to date.",
                    "postprocess_level": postprocess,
                    **result,
                }
            build_result = {
                "status": "ok",
                "build_type": "incremental",
                "summary": (
                    f"Incremental update: {result['files_updated']} files re-parsed, "
                    f"{result['total_nodes']} nodes and "
                    f"{result['total_edges']} edges updated. "
                    f"Changed: {result['changed_files']}. "
                    f"Dependents also updated: {result['dependent_files']}."
                ),
                **result,
            }

        # Pass changed_files for incremental flow/community detection
        changed = result.get("changed_files") if not full_rebuild else None
        warnings = _run_postprocess(
            store, build_result, postprocess,
            full_rebuild=full_rebuild, changed_files=changed,
        )
        if warnings:
            build_result["warnings"] = warnings
        return build_result
    finally:
        store.close()


def run_postprocess(
    flows: bool = True,
    communities: bool = True,
    fts: bool = True,
    repo_root: str | None = None,
) -> dict[str, Any]:
    """Run post-processing steps on an existing graph.

    Useful for running expensive steps (flows, communities) separately
    from the build, or for re-running after the graph has been updated
    with ``postprocess="none"``.

    Args:
        flows: Run flow detection. Default: True.
        communities: Run community detection. Default: True.
        fts: Rebuild FTS index. Default: True.
        repo_root: Repository root path. Auto-detected if omitted.

    Returns:
        Summary of what was computed.
    """
    store, _root = _get_store(repo_root)
    result: dict[str, Any] = {"status": "ok"}
    warnings: list[str] = []

    try:
        # Signatures are always fast — run them
        try:
            rows = store.get_nodes_without_signature()
            sig_updates: list[tuple[str, int]] = []
            for row in rows:
                node_id, name, kind, params, ret = (
                    row[0], row[1], row[2], row[3], row[4],
                )
                if kind in ("Function", "Test"):
                    sig = f"def {name}({params or ''})"
                    if ret:
                        sig += f" -> {ret}"
                elif kind == "Class":
                    sig = f"class {name}"
                else:
                    sig = name
                sig_updates.append((sig[:512], node_id))
            if sig_updates:
                store._conn.executemany(
                    "UPDATE nodes SET signature = ? WHERE id = ?", sig_updates,
                )
            store.commit()
            result["signatures_updated"] = True
        except (sqlite3.OperationalError, TypeError, KeyError) as e:
            logger.warning("Signature computation failed: %s", e)
            warnings.append(f"Signature computation failed: {type(e).__name__}: {e}")

        if fts:
            try:
                from code_review_graph.search import rebuild_fts_index

                fts_count = rebuild_fts_index(store)
                result["fts_indexed"] = fts_count
            except (sqlite3.OperationalError, ImportError) as e:
                logger.warning("FTS index rebuild failed: %s", e)
                warnings.append(f"FTS index rebuild failed: {type(e).__name__}: {e}")

        if flows:
            try:
                from code_review_graph.flows import store_flows as _store_flows
                from code_review_graph.flows import trace_flows as _trace_flows

                traced = _trace_flows(store)
                count = _store_flows(store, traced)
                result["flows_detected"] = count
            except (sqlite3.OperationalError, ImportError) as e:
                logger.warning("Flow detection failed: %s", e)
                warnings.append(f"Flow detection failed: {type(e).__name__}: {e}")

        if communities:
            try:
                from code_review_graph.communities import (
                    detect_communities as _detect_communities,
                )
                from code_review_graph.communities import (
                    store_communities as _store_communities,
                )

                comms = _detect_communities(store)
                count = _store_communities(store, comms)
                result["communities_detected"] = count
            except (sqlite3.OperationalError, ImportError) as e:
                logger.warning("Community detection failed: %s", e)
                warnings.append(f"Community detection failed: {type(e).__name__}: {e}")

        store.set_metadata(
            "last_postprocessed_at", time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        result["summary"] = "Post-processing complete."
        if warnings:
            result["warnings"] = warnings
        return result
    finally:
        store.close()
