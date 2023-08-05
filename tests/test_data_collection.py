# python -m unittest tests.test_data_collection
# test_data_collection.py
import unittest
import sys
import os
import pandas as pd
sys.path.append('./scripts')
from data_collection import DataCollector

class TestDataCollection(unittest.TestCase):
    """The TestDataCollection is a test class for the DataCollector class."""

    def test_add(self):
        """Test the add function."""
        result = 5
        self.assertEqual(result, 5)
        
    def test_fetch_auctions(self):
        """Test the _fetch_auctions method of DataCollector class."""
        # Setup
        notebook_dir = os.getcwd()
        data_dir = os.path.join(notebook_dir, "..", "data")
        data_collector = DataCollector(data_dir)
        data_collector._fetch_auctions()
        data_collector.disconnectDB()
        assert True
        
    def test_fetch_vehicles(self):
        """Test the _fetch_vehicles method of DataCollector class."""
        # Setup
        notebook_dir = os.getcwd()
        data_dir = os.path.join(notebook_dir, "..", "data")
        data_collector = DataCollector(data_dir)
        data_collector._fetch_vehicles()
        data_collector.disconnectDB()
        
    def test_fetch_all(self):
        """Test the fetch_all method of DataCollector class."""
        # Setup
        notebook_dir = os.getcwd()
        data_dir = os.path.join(notebook_dir, "..", "data")
        data_collector = DataCollector(data_dir)
        data_collector.fetch_all()
        data_collector.disconnectDB()
        
    def test_fetch_inspections(self):
        """Test the _fetch_inspections method of DataCollector class."""
        # Setup
        notebook_dir = os.getcwd()
        data_dir = os.path.join(notebook_dir, "..", "data")
        data_collector = DataCollector(data_dir)
        data_collector._fetch_inspections()
        data_collector.disconnectDB()

if __name__ == '__main__':
    unittest.main()
