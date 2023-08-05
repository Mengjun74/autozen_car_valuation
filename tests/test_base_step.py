# python -m unittest tests.test_base_step
# test_base_step.py
import sys
import os
import pandas as pd
import unittest
from unittest.mock import patch, MagicMock
sys.path.append('./scripts')
from base_step import PipelineStep  # replace `your_module` with the name of the module that contains PipelineStep

class TestPipelineStep(unittest.TestCase):
    @patch("os.getenv")
    def test_get_db_creds_from_env(self, mock_getenv):
        mock_getenv.return_value = "test"
        pipeline = PipelineStep("test_dir")
        self.assertDictEqual(pipeline.db_creds, {'user': 'test', 'password': 'test', 'host': 'test', 'database': 'test'})

    @patch("os.getenv")
    @patch("json.load")
    def test_get_db_creds_from_file(self, mock_json_load, mock_getenv):
        mock_getenv.return_value = None
        mock_json_load.return_value = {'user': 'file_test', 'password': 'file_test', 'host': 'file_test', 'database': 'file_test'}
        pipeline = PipelineStep("test_dir")
        self.assertDictEqual(pipeline.db_creds, {'user': 'file_test', 'password': 'file_test', 'host': 'file_test', 'database': 'file_test'})

    def test_progress_next(self):
        pipeline = PipelineStep("test_dir")
        pipeline.progressNext("test_message")
        self.assertEqual(pipeline.progress_bar.suffix, "test_message")
        
    @patch("mysql.connector.connect")
    def test_reconnect_db(self, mock_connect):
        mock_connect.return_value = MagicMock()
        pipeline = PipelineStep("test_dir")
        pipeline.reconnectDB()
        self.assertIsNotNone(pipeline.cnx)

    @patch("torch.save")
    def test_save_nn_model(self, mock_save):
        pipeline = PipelineStep("test_dir")
        pipeline.save_nn_model(MagicMock(), "test_model")
        mock_save.assert_called_once()

if __name__ == "__main__":
    unittest.main()