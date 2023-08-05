# data_preprocessing.py
from base_step import PipelineStep
from base_step import VERBOSE
import pandas as pd
import numpy
import json
import os
import threading

class DataPreprocessor (PipelineStep):
    """
    A class to perform data preprocessing on the Autozen dataset. Inherits from `PipelineStep`.

    Attributes:
        data_dir (str): The directory where the data files are located.
    """
    def __init__(self, data_dir):
        super().__init__(data_dir, None, 
                         num_of_steps=6, 
                         progress_message='DataProcessor: processing...')
        self._set_input_output_paths()

    def _set_input_output_paths(self):
        """
        Sets the input and output file paths for data processing.
        """
        # Input
        self.INSPECTIONS_RAW_FILE = os.path.join(self.data_dir, "raw_az_inspections.csv")
        self.VEHICLES_RAW_FILE = os.path.join(self.data_dir, "raw_az_vehicles.csv")
        self.AUCTIONS_RAW_FILE = os.path.join(self.data_dir, "raw_az_auctions.csv")
        self.INSPECTIONSCHEDULE_RAW_FILE = os.path.join(self.data_dir, "raw_az_inspection_schedules.csv")
        # Output
        self.INSPECTIONS_PROCESSED_FILE = os.path.join(self.data_dir, "processed_az_inspections.csv") 
        self.MERGED_PROCESSED_WON = os.path.join(self.data_dir, "processed_az_auctioned_won.csv")
        self.MERGED_PROCESSED_AUCTIONED = os.path.join(self.data_dir, "processed_az_auctioned.csv")
        self.INSPECTIONSCHEDULE_PROCESSED_FILE = os.path.join(self.data_dir, "processed_az_inspection_schedules.csv") 

    def printHelp(self):
        """
        Prints the input and output file paths for data processing.
        """
        message = f"""
        DataProcessor input:
            {self.INSPECTIONS_RAW_FILE} 
            {self.VEHICLES_RAW_FILE} 
            {self.AUCTIONS_RAW_FILE} 
            {self.INSPECTIONSCHEDULE_RAW_FILE}
        DataProcessor output:
            {self.INSPECTIONS_PROCESSED_FILE} 
            {self.MERGED_PROCESSED_WON} 
            {self.MERGED_PROCESSED_AUCTIONED} 
            {self.INSPECTIONSCHEDULE_PROCESSED_FILE}
        """
        print(message)

    def _flatten_inspections(self,df, json_col, to_drop_set):
        """
        Flattens the inspections data by creating new columns from the specified JSON column.
        """    
        MISSING_TEXT = numpy.nan
        MISSING_NUMBER = numpy.nan
        MISSING_LIST = numpy.nan
        MISSING_PART_STATUS = numpy.nan
        VALUE_FILE = numpy.nan
        BASE_MISSING = numpy.nan
        
        # Create 4 sets 
        ordinal_features = set()
        numeric_features = set()

        
        # Check if the column exists in the DataFrame
        if json_col not in df.columns:
            raise ValueError(f"Column '{json_col}' not found in DataFrame")

        # Initialize a list to store the dictionaries
        new_data = []

        # Iterate through each row in the DataFrame
        for index in range(len(df)):
            # Load the JSON data from the specified column
            json_data_str = df.iloc[index][json_col]
            if isinstance(json_data_str, float):
                continue
            json_data = json.loads(json_data_str)

            # Initialize a dictionary to store new columns and values
            new_cols = {}

            # Iterate through each JSON object and create new columns
            for item in json_data:
                new_col = item['name']
                item_type = item['type']
                if new_col in to_drop_set:
                    continue
                if new_col.endswith('Comment'):
                    continue
                if new_col.endswith('Note'):
                    continue
                if new_col.endswith('Photos') or new_col.endswith('Photo'):
                    continue
                if new_col.find('comment') != -1:
                    continue
                if new_col.find('photo') != -1:
                    continue
                if new_col is None or item_type is None:
                    continue
                if "text" == item_type:
                    # that's a text field
                    if 'value' in item:
                        val = item['value']
                    else:
                        val = MISSING_TEXT
                elif "file" == item_type:
                    # shortcut for file
                    val = VALUE_FILE
                elif "part_status" == item_type:
                    ordinal_features.add(json_col+'_'+new_col)
                    if 'value' in item :
                        if item['value'] is None:
                            val = MISSING_PART_STATUS
                        else:
                            if 'status' in item['value']:
                                val = item['value']['status']
                            else:
                                val = MISSING_PART_STATUS
                    else:
                        val = MISSING_PART_STATUS
                elif "number" == item_type:
                    numeric_features.add(json_col+'_'+new_col)
                    if 'value' in item:
                        val = item['value']
                    else:
                        val = MISSING_NUMBER
                elif "list" == item_type:
                    if 'value' in item:
                        val = item['value']
                        if isinstance(val, dict) and 'value' in val:
                            val = val['value']
                    else:
                        val = MISSING_LIST
                else:
                    val = BASE_MISSING
                if '{}' == val:
                    val = numpy.nan
                new_cols[json_col+'_'+new_col] = val

            # Add the new columns dictionary to the list
            new_data.append(new_cols)

        # Create a new DataFrame from the list of dictionaries and concatenate it with the original DataFrame
        new_df = pd.DataFrame(new_data)
        df = pd.concat([df, new_df], axis=1)
        return df, list(ordinal_features), list(numeric_features)


    def _preprocess_inspections(self):
        """
        Preprocesses the inspections data by flattening JSON columns and saving the processed data to a new CSV file.
        """
        self.progressNext(f'DataProcessor: processing {self.INSPECTIONS_PROCESSED_FILE}')
        COLUMNS = ['id', 'general', 
           'outside', 'tires', 'under_vehicle', 'under_the_hood', 
           'lighting', 'interior', 'test_drive', 'created_at']

        COLUMNS_TO_DROP_SET = set(['inspector_id','schedule_id', 'documents', 'inspection_code',
                    'milage','gaugeCluster','PhotoOfVIN',
                   'PhotoOfLabel', 'generalNote','emissionsSticker',
                   'dashWarningLightsNotes', 'warrantyDocumentPhoto', 
                   'lastVehicleService', 'mileageUnit', 'speedometerOdometerConfigurable',
                   'engineBayPhoto','exteriorRoofPhoto','driverRearTireMakeSizeDate',
                   'driverFrontTireMakeSizeDate', 'passengerRearTireMakeSizeDate',
                   'passengerFrontTireMakeSizeDate', 'aftermarketWheelsTireMakeSizeDate',
                   'extraSetOfWheelsTiresMakeSizeDate', 'trunkLiner','Unamed:0'])
        # Read the CSV file into a DataFrame
        inspections_raw_df = pd.read_csv(self.INSPECTIONS_RAW_FILE)

        inspections_raw_df.head()
        flatten_df = inspections_raw_df
        ordinal_features_list = []
        numeric_features_list = []
        for column in COLUMNS[1:-1]:
            flatten_df , l , n = self._flatten_inspections(flatten_df, column, COLUMNS_TO_DROP_SET)
            ordinal_features_list += l
            numeric_features_list += n
            if VERBOSE:
                print(f"after processing group '{column}': shape is {flatten_df.shape}")
            
        flatten_df.drop(COLUMNS[1:-1], axis=1, inplace=True)
        if VERBOSE:
            print(f"after dropping JSON columns: shape is {flatten_df.shape}")

        # Save to CSV
        flatten_df.to_csv(self.INSPECTIONS_PROCESSED_FILE, index=False)
        self.progressNext(f'DataProcessor: processed {self.INSPECTIONS_PROCESSED_FILE}')

    def _join_auctions_vehicles_inspections(self):
        """
        Joins the auctions, vehicles, and inspections tables based on common identifiers and saves the merged data to a new CSV file.
        """
        self.progressNext(f'DataProcessor: cleaning {self.MERGED_PROCESSED_AUCTIONED}')
        # read dfs
        inspections = pd.read_csv(self.INSPECTIONS_PROCESSED_FILE, low_memory=False)
        auctions = pd.read_csv(self.AUCTIONS_RAW_FILE)
        vehicles = pd.read_csv(self.VEHICLES_RAW_FILE, index_col=0)
        # join
        auctions_productID = auctions.product_id.to_list()
        inspections_ID = inspections.general_vinNumber.to_list()
        vehicle_ID = vehicles.id.to_list()
        vehicle_vin = vehicles.vehicle_vin.to_list()

        # Convert lists to sets
        auctions_set = set(auctions_productID)
        inspections_set = set(inspections_ID)
        vehicle_set = set(vehicle_ID)
        vehicle_vin_set = set(vehicle_vin)

        # Find common elements between sets
        common_elements1 = auctions_set.intersection(vehicle_set)

        # Print the number of common elements
        if VERBOSE:
            print("Number of common elements between auction product ID and vehicle ID:", len(common_elements1))

        # Find common elements between sets
        common_elements2 = inspections_set.intersection(vehicle_vin_set)

        if VERBOSE:
            # Print the number of common elements
            print("Number of common elements between inspection VIN and vehicle VIN:", len(common_elements2))
            # Print the shape of the three datasets 
            print("Shape of auctions:", auctions.shape)
            print("Shape of vehicle:", vehicles.shape)
            print("Shape of inspections:", inspections.shape)

        # Merge auctions and inspections tables based on product_id and vinNumber
        auctions_vehicle = auctions.merge(vehicles, how='inner', left_on='product_id', right_on='id')

        if VERBOSE:
            # Number of shared elements
            print("Number of shared elements", auctions_vehicle.vehicle_vin.unique().shape)

            # Shape of merged df
            print("Shape of merged df", auctions_vehicle.shape)

        # Merge vehicle table with auctions_inspections table based on id and vehicle_vin
        merged_df = inspections.merge(auctions_vehicle, how='inner', left_on='general_vinNumber', right_on='vehicle_vin')

        #### Engineered features 
        # 1- Car Age
        import datetime
        merged_df['general_year'] = merged_df['general_year'].fillna(2000).astype(int)
        current_year = datetime.datetime.now().year
        merged_df['car_age'] = (current_year - merged_df['general_year']).astype(int)
        # 2- Month of Year
        merged_df['created_at'] = pd.to_datetime(merged_df['created_at'])
        merged_df['month_of_year'] = merged_df['created_at'].dt.month

        if VERBOSE:
            # Number of shared elements
            print("Number of shared elements", merged_df.vehicle_vin.unique().shape)

            # Shape of merged df
            print("Shape of merged df", merged_df.shape)

        # only keeping the sold cars
        merged_df_bid_won = merged_df.query('bid_winning > 0')
        filtered_merged_df_bid_won = merged_df_bid_won.sort_values(by='bid_winning', ascending=False)
        filtered_merged_df_bid_won.drop_duplicates(subset='vehicle_vin', inplace=True)
        filtered_merged_df_bid_won

        filtered_merged_df_bid_won.to_csv(self.MERGED_PROCESSED_WON)
        merged_df.drop_duplicates(subset='vehicle_vin', inplace=True)
        merged_df.to_csv(self.MERGED_PROCESSED_AUCTIONED)
        self.progressNext(f'DataProcessor: processed {self.MERGED_PROCESSED_AUCTIONED}')

    def _preprocess_inspections_schedules(self):
        """
        Preprocesses the inspection schedules data by flattening JSON columns and saving the processed data to a new CSV file.
        """        
        self.progressNext(f'DataProcessor: processing {self.INSPECTIONSCHEDULE_RAW_FILE}')
        COLUMNS_TO_DROP_SET = set(['id','location'])
        # Read the CSV file into a DataFrame
        inspection_schedules_raw_df = pd.read_csv(self.INSPECTIONSCHEDULE_RAW_FILE)
        flatten_df = inspection_schedules_raw_df
        flatten_df = self._flatten_inspection_schedules(flatten_df, 'location', COLUMNS_TO_DROP_SET)
        flatten_df.drop(COLUMNS_TO_DROP_SET, axis=1, inplace=True)
        if VERBOSE:
            print(f"after dropping JSON columns: shape is {flatten_df.shape}")    
            
        # Save to CSV
        flatten_df.to_csv(self.INSPECTIONSCHEDULE_PROCESSED_FILE, index=False)
        self.progressNext(f'DataProcessor: processed {self.INSPECTIONSCHEDULE_PROCESSED_FILE}')
        
    def _flatten_inspection_schedules(self,df, json_col, to_drop_set):
        BASE_MISSING = numpy.nan
        columns_to_keep = {'provinceCode','cityName'}
        # Check if the column exists in the DataFrame
        if json_col not in df.columns:
            raise ValueError(f"Column '{json_col}' not found in DataFrame")
        # Initialize a list to store the dictionaries
        new_data = []

        # Iterate through each row in the DataFrame
        for index in range(len(df)):
            # Load the JSON data from the specified column
            json_data_str = df.iloc[index][json_col]
            if isinstance(json_data_str, float):
                if VERBOSE:
                    print(f"Skipping row {index} because it is a float in col {json_col}")
                continue
            json_data = json.loads(json_data_str)

            # Initialize a dictionary to store new columns and values
            new_cols = {}

            # Iterate through each JSON object and create new columns
            for item in json_data:
                new_col = item
                val = json_data[item]
                if new_col in columns_to_keep:
                    new_cols[new_col] = val
                else:
                    continue
            # Add the new columns dictionary to the list
            new_data.append(new_cols)

        # Create a new DataFrame from the list of dictionaries and concatenate it with the original DataFrame
        new_df = pd.DataFrame(new_data)
        df = pd.concat([df, new_df], axis=1)
        return df
    
    def process_all(self):
        self._preprocess_inspections()
        self._preprocess_inspections_schedules()
        self._join_auctions_vehicles_inspections()


if __name__ == '__main__':
    notebook_dir = os.getcwd()
    data_dir = os.path.join(notebook_dir, ".", "data")
    data_preprocessor = DataPreprocessor(data_dir)
    data_preprocessor.process_all()
