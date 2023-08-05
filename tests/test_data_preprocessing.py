# python -m unittest tests.test_data_preprocessing
# test_data_collection.py
import unittest
import os
import pandas as pd
import sys
sys.path.append('./scripts')
from data_preprocessing import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):

    def setUp(self):
        """
        Sets up the test fixture before exercising it.
        """
        self.notebook_dir = os.getcwd()
        self.data_dir = os.path.join(self.notebook_dir, ".", "data")
        self.data_preprocessor = DataPreprocessor(self.data_dir)

    def test_set_input_output_paths(self):
        """
        Tests that the _set_input_output_paths function correctly sets the file paths.
        """
        self.data_preprocessor._set_input_output_paths()

        # Assert the file paths are correctly set
        self.assertEqual(self.data_preprocessor.INSPECTIONS_RAW_FILE, os.path.join(self.data_dir, "raw_az_inspections.csv"))
        self.assertEqual(self.data_preprocessor.VEHICLES_RAW_FILE, os.path.join(self.data_dir, "raw_az_vehicles.csv"))
        # Continue for the rest of the files

    def test_preprocess_inspections(self):
        """
        Tests that the _preprocess_inspections function correctly processes the inspections data.
        """
        self.data_preprocessor._preprocess_inspections()
        
        # Check if the processed file exists and is not empty
        self.assertTrue(os.path.exists(self.data_preprocessor.INSPECTIONS_PROCESSED_FILE))
        self.assertGreater(os.stat(self.data_preprocessor.INSPECTIONS_PROCESSED_FILE).st_size, 0)

    def test_preprocess_inspections_schedules(self):
        """
        Tests that the _preprocess_inspections_schedules function correctly processes the inspections schedules data.
        """
        self.data_preprocessor._preprocess_inspections_schedules()

        # Check if the processed file exists and is not empty
        self.assertTrue(os.path.exists(self.data_preprocessor.INSPECTIONSCHEDULE_PROCESSED_FILE))
        self.assertGreater(os.stat(self.data_preprocessor.INSPECTIONSCHEDULE_PROCESSED_FILE).st_size, 0)

    def test_join_auctions_vehicles_inspections(self):
        """
        Tests that the _join_auctions_vehicles_inspections function correctly joins the auctions, vehicles, and inspections tables.
        """
        self.data_preprocessor._join_auctions_vehicles_inspections()

        # Check if the processed files exist and are not empty
        self.assertTrue(os.path.exists(self.data_preprocessor.MERGED_PROCESSED_AUCTIONED))
        self.assertGreater(os.stat(self.data_preprocessor.MERGED_PROCESSED_AUCTIONED).st_size, 0)
        self.assertTrue(os.path.exists(self.data_preprocessor.MERGED_PROCESSED_WON))
        self.assertGreater(os.stat(self.data_preprocessor.MERGED_PROCESSED_WON).st_size, 0)

    def test_flatten_inspections(self):
        """
        Tests that the _flatten_inspections function correctly flattens the inspections data.
        """
        # Create a sample DataFrame with a JSON column
        df = pd.DataFrame({"general": ['{"name": "sample1", "type": "text", "value": "test"}']})
        df, _, _ = self.data_preprocessor._flatten_inspections(df, "general", set())

        # Assert the DataFrame has the new column and the value is correct
        self.assertIn('general_sample1', df.columns)
        self.assertEqual(df.at[0, 'general_sample1'], 'test')

