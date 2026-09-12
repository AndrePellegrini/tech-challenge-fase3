"""Testes offline do acesso à Base dos Dados e da configuração do projeto."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.preprocessing import bigquery_reader as reader


class ReaderTests(unittest.TestCase):
    def test_loads_env_from_project_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "env").write_text("BILLING_PROJECT_ID=synthetic-project\n")
            with patch.object(reader, "PROJECT_ROOT", root), patch.dict(os.environ, {}, clear=True), patch.object(reader.bd, "read_sql", return_value=pd.DataFrame({"ok": [1]})) as read:
                result = reader.run_query("SELECT 1 AS ok")
                read.assert_called_once_with(query="SELECT 1 AS ok", billing_project_id="synthetic-project")
                self.assertEqual(result.iloc[0, 0], 1)

    def test_explicit_project_remains_compatible(self):
        with patch.object(reader, "load_dotenv"), patch.object(reader.bd, "read_sql") as read:
            reader.run_query("SELECT 1", project_id="explicit-project")
            read.assert_called_once_with(query="SELECT 1", billing_project_id="explicit-project")

    def test_environment_has_priority_over_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "env").write_text("BILLING_PROJECT_ID=file-project\n")
            with patch.object(reader, "PROJECT_ROOT", root), patch.dict(os.environ, {"BILLING_PROJECT_ID": "environment-project"}, clear=True), patch.object(reader.bd, "read_sql") as read:
                reader.run_query("SELECT 1")
                self.assertEqual(read.call_args.kwargs["billing_project_id"], "environment-project")

    def test_missing_project_fails_before_query(self):
        with patch.object(reader, "load_dotenv"), patch.dict(os.environ, {}, clear=True), patch.object(reader.bd, "read_sql") as read:
            with self.assertRaisesRegex(ValueError, "BILLING_PROJECT_ID"):
                reader.run_query("SELECT 1")
            read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
