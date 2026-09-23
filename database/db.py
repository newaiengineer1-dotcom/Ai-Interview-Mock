def delete_session(session_id):
    """Delete a session and all its turns."""
    with _conn() as c:
        c.execute("DELETE FROM turns WHERE session_id=?", (session_id,))
        c.execute("DELETE FROM sessions WHERE id=?", (session_id,))
    return True


def delete_all_sessions():
    """Wipe all sessions and turns."""
    with _conn() as c:
        c.execute("DELETE FROM turns")
        c.execute("DELETE FROM sessions")
    return True


def get_session_stats():
    """Return aggregate stats for the dashboard."""
    with _conn() as c:
        row = c.execute("""
            SELECT COUNT(*) as total,
                   COALESCE(AVG(avg_score), 0) as avg_score,
                   COALESCE(MAX(avg_score), 0) as best_score,
                   COALESCE(MIN(avg_score), 0) as worst_score
            FROM sessions WHERE avg_score > 0
        """).fetchone()
        return dict(row) if row else {
            "total": 0, "avg_score": 0, "best_score": 0, "worst_score": 0}
