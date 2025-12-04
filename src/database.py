#!/usr/bin/env python3
"""
Database module for Atlas Pipeline History.

Provides MySQL connection management and CRUD operations for pipeline run history.
"""

import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Optional

import pymysql
from pymysql.cursors import DictCursor
from dotenv import load_dotenv

load_dotenv()


class PipelineHistoryDB:
    """Manages MySQL database operations for pipeline run history."""

    def __init__(self):
        """Initialize database connection settings from environment variables."""
        self.host = os.getenv("MYSQL_HOST", "localhost")
        self.port = int(os.getenv("MYSQL_PORT", 3306))
        self.user = os.getenv("MYSQL_USER", "root")
        self.password = os.getenv("MYSQL_PASSWORD", "")
        self.database = os.getenv("MYSQL_DATABASE", "atlas")

        self._ensure_database_exists()
        self._ensure_tables_exist()

    @contextmanager
    def get_connection(self, use_database: bool = True):
        """Context manager for database connections."""
        connection = pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database if use_database else None,
            charset="utf8mb4",
            cursorclass=DictCursor,
            autocommit=True,
        )
        try:
            yield connection
        finally:
            connection.close()

    def _ensure_database_exists(self) -> None:
        """Create database if it doesn't exist."""
        try:
            with self.get_connection(use_database=False) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        f"CREATE DATABASE IF NOT EXISTS `{self.database}` "
                        f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                    )
        except Exception as e:
            print(f"[DB] Error creating database: {e}")
            raise

    def _ensure_tables_exist(self) -> None:
        """Create tables if they don't exist."""
        create_pipeline_runs_sql = """
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            run_id VARCHAR(64) NOT NULL UNIQUE,
            search_query VARCHAR(1000) NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            video_count INT DEFAULT 0,
            transcript_count INT DEFAULT 0,
            summary_count INT DEFAULT 0,
            output_folder_path VARCHAR(500) NOT NULL,
            status ENUM('running', 'success', 'failed', 'partial') DEFAULT 'running',
            duration_seconds FLOAT DEFAULT NULL,
            error_message TEXT DEFAULT NULL,
            config_json TEXT DEFAULT NULL,
            user_email VARCHAR(255) DEFAULT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_timestamp (timestamp DESC),
            INDEX idx_status (status),
            INDEX idx_search_query (search_query(255)),
            INDEX idx_user_email (user_email)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """

        create_user_logs_sql = """
        CREATE TABLE IF NOT EXISTS user_activity_logs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_email VARCHAR(255) NOT NULL,
            user_name VARCHAR(255) DEFAULT NULL,
            action VARCHAR(100) NOT NULL,
            details TEXT DEFAULT NULL,
            ip_address VARCHAR(45) DEFAULT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_user_email (user_email),
            INDEX idx_action (action),
            INDEX idx_timestamp (timestamp DESC)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(create_pipeline_runs_sql)
                    cursor.execute(create_user_logs_sql)
        except Exception as e:
            print(f"[DB] Error creating tables: {e}")
            raise

    def create_run(
        self,
        search_query: str,
        output_folder_path: str,
        config: Dict,
        user_email: Optional[str] = None,
    ) -> str:
        """
        Create a new pipeline run record.

        Args:
            search_query: The search query used
            output_folder_path: Path to the output folder
            config: Configuration dict with max_videos, language, workers, etc.
            user_email: Email of the user who initiated the run

        Returns:
            run_id: Unique identifier for this run
        """
        run_id = str(uuid.uuid4())
        config_json = json.dumps(config)

        sql = """
        INSERT INTO pipeline_runs
        (run_id, search_query, output_folder_path, config_json, status, user_email)
        VALUES (%s, %s, %s, %s, 'running', %s)
        """

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        sql, (run_id, search_query, output_folder_path, config_json, user_email)
                    )
            print(f"[DB] Created run: {run_id} by {user_email or 'unknown'}")
            return run_id
        except Exception as e:
            print(f"[DB] Error creating run: {e}")
            raise

    def update_run_progress(
        self,
        run_id: str,
        video_count: Optional[int] = None,
        transcript_count: Optional[int] = None,
        summary_count: Optional[int] = None,
    ) -> bool:
        """
        Update counts during pipeline execution.

        Args:
            run_id: The run identifier
            video_count: Number of videos found (optional)
            transcript_count: Number of transcripts fetched (optional)
            summary_count: Number of summaries generated (optional)

        Returns:
            bool: True if update was successful
        """
        updates = []
        values = []

        if video_count is not None:
            updates.append("video_count = %s")
            values.append(video_count)
        if transcript_count is not None:
            updates.append("transcript_count = %s")
            values.append(transcript_count)
        if summary_count is not None:
            updates.append("summary_count = %s")
            values.append(summary_count)

        if not updates:
            return True

        sql = f"UPDATE pipeline_runs SET {', '.join(updates)} WHERE run_id = %s"
        values.append(run_id)

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, values)
            return True
        except Exception as e:
            print(f"[DB] Error updating run progress: {e}")
            return False

    def complete_run(
        self,
        run_id: str,
        status: str,
        duration_seconds: float,
        error_message: Optional[str] = None,
    ) -> bool:
        """
        Mark a run as complete.

        Args:
            run_id: The run identifier
            status: Final status ('success', 'failed', 'partial')
            duration_seconds: Total execution time
            error_message: Error details if failed (optional)

        Returns:
            bool: True if update was successful
        """
        sql = """
        UPDATE pipeline_runs
        SET status = %s, duration_seconds = %s, error_message = %s
        WHERE run_id = %s
        """

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, (status, duration_seconds, error_message, run_id))
            print(f"[DB] Completed run {run_id} with status: {status}")
            return True
        except Exception as e:
            print(f"[DB] Error completing run: {e}")
            return False

    def get_run(self, run_id: str) -> Optional[Dict]:
        """
        Get a single run by ID.

        Args:
            run_id: The run identifier

        Returns:
            Dict with run details or None if not found
        """
        sql = "SELECT * FROM pipeline_runs WHERE run_id = %s"

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, (run_id,))
                    return cursor.fetchone()
        except Exception as e:
            print(f"[DB] Error getting run: {e}")
            return None

    def get_all_runs(
        self,
        limit: int = 50,
        offset: int = 0,
        status_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Get all runs with pagination.

        Args:
            limit: Maximum number of runs to return
            offset: Number of runs to skip
            status_filter: Filter by status (optional)

        Returns:
            List of run dictionaries
        """
        if status_filter:
            sql = """
            SELECT * FROM pipeline_runs
            WHERE status = %s
            ORDER BY timestamp DESC
            LIMIT %s OFFSET %s
            """
            params = (status_filter, limit, offset)
        else:
            sql = """
            SELECT * FROM pipeline_runs
            ORDER BY timestamp DESC
            LIMIT %s OFFSET %s
            """
            params = (limit, offset)

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, params)
                    return cursor.fetchall()
        except Exception as e:
            print(f"[DB] Error getting runs: {e}")
            return []

    def delete_run(self, run_id: str) -> bool:
        """
        Delete a run record.

        Args:
            run_id: The run identifier

        Returns:
            bool: True if deletion was successful
        """
        sql = "DELETE FROM pipeline_runs WHERE run_id = %s"

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, (run_id,))
            print(f"[DB] Deleted run: {run_id}")
            return True
        except Exception as e:
            print(f"[DB] Error deleting run: {e}")
            return False

    def get_run_count(self, status_filter: Optional[str] = None) -> int:
        """
        Get total count of runs.

        Args:
            status_filter: Filter by status (optional)

        Returns:
            Total count of runs
        """
        if status_filter:
            sql = "SELECT COUNT(*) as count FROM pipeline_runs WHERE status = %s"
            params = (status_filter,)
        else:
            sql = "SELECT COUNT(*) as count FROM pipeline_runs"
            params = ()

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, params)
                    result = cursor.fetchone()
                    return result["count"] if result else 0
        except Exception as e:
            print(f"[DB] Error getting run count: {e}")
            return 0

    # ==================== User Activity Logging ====================

    def log_user_action(
        self,
        user_email: str,
        action: str,
        details: Optional[str] = None,
        user_name: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> bool:
        """
        Log a user action.

        Args:
            user_email: User's email address
            action: Action type (e.g., 'login', 'search', 'view_history', 'delete_run')
            details: Additional details about the action
            user_name: User's display name
            ip_address: User's IP address

        Returns:
            bool: True if logging was successful
        """
        sql = """
        INSERT INTO user_activity_logs
        (user_email, user_name, action, details, ip_address)
        VALUES (%s, %s, %s, %s, %s)
        """

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, (user_email, user_name, action, details, ip_address))
            print(f"[DB] Logged action: {action} by {user_email}")
            return True
        except Exception as e:
            print(f"[DB] Error logging user action: {e}")
            return False

    def get_user_logs(
        self,
        user_email: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict]:
        """
        Get user activity logs with optional filters.

        Args:
            user_email: Filter by user email (optional)
            action: Filter by action type (optional)
            limit: Maximum number of logs to return
            offset: Number of logs to skip

        Returns:
            List of log dictionaries
        """
        conditions = []
        params = []

        if user_email:
            conditions.append("user_email = %s")
            params.append(user_email)
        if action:
            conditions.append("action = %s")
            params.append(action)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        sql = f"""
        SELECT * FROM user_activity_logs
        {where_clause}
        ORDER BY timestamp DESC
        LIMIT %s OFFSET %s
        """
        params.extend([limit, offset])

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, params)
                    return cursor.fetchall()
        except Exception as e:
            print(f"[DB] Error getting user logs: {e}")
            return []

    def get_user_log_count(
        self, user_email: Optional[str] = None, action: Optional[str] = None
    ) -> int:
        """Get total count of user logs with optional filters."""
        conditions = []
        params = []

        if user_email:
            conditions.append("user_email = %s")
            params.append(user_email)
        if action:
            conditions.append("action = %s")
            params.append(action)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        sql = f"SELECT COUNT(*) as count FROM user_activity_logs {where_clause}"

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, params)
                    result = cursor.fetchone()
                    return result["count"] if result else 0
        except Exception as e:
            print(f"[DB] Error getting user log count: {e}")
            return 0


def get_db_or_none() -> Optional[PipelineHistoryDB]:
    """
    Attempt to get database connection, return None on failure.

    This allows the app to work without database if MySQL is not configured.
    """
    try:
        return PipelineHistoryDB()
    except Exception as e:
        print(f"[DB] Connection failed: {e}")
        print("[DB] Pipeline history will not be recorded.")
        return None
