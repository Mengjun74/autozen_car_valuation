# data_collection.py
from base_step import PipelineStep
from base_step import VERBOSE
import os
import mysql.connector
import pandas as pd
from progress.bar import ChargingBar

class DataCollector (PipelineStep):
    """
    A class for fetching and storing data from the Autozen relational database. Inherits from `PipelineStep`.
    """
    def __init__(self, data_dir, db_creds=None):
        """
        Initialize a DataCollector instance with the specified data directory.
        """
        super().__init__(data_dir, db_creds,8,'DataCollector: fetching...')
        self.AUCTIONS_RAW_FILE = os.path.join(self.data_dir,"raw_az_auctions.csv")
        self.VEHICLES_RAW_FILE = os.path.join(self.data_dir,"raw_az_vehicles.csv")
        self.INSPECTIONS_RAW_FILE = os.path.join(self.data_dir,"raw_az_inspections.csv")
        self.INSPECTIONSCHEDULE_RAW_FILE = os.path.join(self.data_dir,"raw_az_inspection_schedules.csv")
        self.reconnectDB()

    def printHelp(self):
        """
        Prints the file paths where the fetched data is stored.
        """
        message = f"""
        DataCollector input:
            {self.AUCTIONS_RAW_FILE} 
            {self.VEHICLES_RAW_FILE} 
            {self.INSPECTIONS_RAW_FILE} 
            {self.INSPECTIONSCHEDULE_RAW_FILE}
        """
        print(message)

    def _fetch_data(self, table_name, columns, db_name, file_name, index=False):
        """
        Fetches data from a specified table and saves it to a CSV file.

        :param table_name: The name of the table in the database from which to fetch data.
        :param columns: The columns to select in the SQL query.
        :param db_name: The name of the database to use.
        :param file_name: The name of the file to which the data should be written.
        """
        self.progressNext(f'DataCollector: fetching {file_name}')
        SQL_QUERY = f"SELECT {', '.join(columns)} FROM {table_name}"
        
        cursor = self.cnx.cursor()
        cursor.execute(f"USE `{db_name}`")
        cursor.execute(SQL_QUERY)
        
        results = cursor.fetchall()
        df = pd.DataFrame(results, columns=[desc[0] for desc in cursor.description])
        cursor.close()
        
        df.to_csv(file_name, index=index)
        self.progressNext(f'DataCollector: fetched {file_name}')
        

    def _fetch_auctions(self):
        """Fetches auction data and saves it to a CSV file."""
        columns = [
            'id',
            'status',
            'product_id',
            'bid_winning',
            'bid_start',
            'created_at'
        ]
        self._fetch_data("auctions", columns, "autozen", self.AUCTIONS_RAW_FILE)

    def _fetch_vehicles(self):
        """Fetches vehicle data and saves it to a CSV file."""
        columns = [
            'id',
            'vehicle_vin',
            'vehicle_mileage',
            'vehicle_lead_source',
            'seller_type',
            'num_auctions',
            'valuation_autozen_low',
            'valuation_autozen_high',
            'valuation_cbb_trade_in_low',
            'valuation_cbb_trade_in_high',
            'valuation_cbb_base_retail_avg',
            'valuation_highest_bid_amount',
            'valuation_seller_asking_price',
            'valuation_starting_price',
        ]
        self._fetch_data("vehicles", columns, "autozen-analytics", self.VEHICLES_RAW_FILE,index=True)


    def _fetch_inspections(self):
        """Fetches inspection data and saves it to a CSV file."""
        columns = [
            'id', 
            'general', 
            'outside', 
            'tires', 
            'under_vehicle', 
            'under_the_hood', 
            'lighting', 
            'interior', 
            'test_drive'
        ]
        self._fetch_data("inspections", columns, "autozen", self.INSPECTIONS_RAW_FILE)

    def _fetch_inspection_schedules(self):
        """Fetches inspection schedule data and saves it to a CSV file."""
        columns = ['id', 'product_id', 'location']
        self._fetch_data("inspection_schedules", columns, "autozen", self.INSPECTIONSCHEDULE_RAW_FILE)

    def fetch_all(self):
        """
        Fetches all data from the database.
        """
        for fetch_method in [self._fetch_inspection_schedules, 
                             self._fetch_auctions, 
                             self._fetch_inspections, 
                             self._fetch_vehicles]:
            fetch_method()

if __name__ == '__main__':
    notebook_dir = os.getcwd()
    data_dir = os.path.join(notebook_dir, ".", "data")
    data_collector = DataCollector(data_dir)
    data_collector.fetch_all()
    data_collector.disconnectDB()