from sklearn.model_selection import cross_validate, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
import pandas as pd
from utils import assert_disjoint_sets, subtract_subsets_from_superset

class HandleUnknownTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, method='unknown', fill_value='Unknown'):
        self.method = method
        self.fill_value = fill_value

    def fit(self, X, y=None):
        self.columns_ = X.columns
        if self.method == 'most_frequent':
            self.fill_ = X.mode().loc[0]
        else:
            self.fill_ = pd.Series([self.fill_value] * X.shape[1], index=self.columns_)
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        for col in self.columns_:
            if col in X_.columns:
                if self.fill_[col] not in X_[col].unique():
                    X_.loc[~X_[col].isin(self.fill_), col] = self.fill_[col]
            else:
                X_[col] = pd.Series([self.fill_[col]] * X_.shape[0])
        return X_



class AutozenFeatures: 
    """
    A class to define the specific features used in the Autozen model. 
    These features are categorized based on their types: numeric, ordinal, categorical, binary and to be dropped features.

    Attributes:
    -----------
    numeric_feat_list : list
        A list of features representing numerical variables in the data.

    ordinal_feat_list : list
        A list of features representing ordinal variables in the data.

    categorical_feat_list : list
        A list of features representing categorical variables in the data.

    ordinal_feat_list_a : list
        A list of specific features representing ordinal variables in the data.

    ordinal_feat_list_b : list
        A list of specific features representing ordinal variables in the data.

    binary_feat_list : list
        A list of features representing binary variables in the data.

    to_drop_feat_list : list
        A list of features to be excluded from the model.
    """
    
    @staticmethod 
    def validate():
        assert_disjoint_sets([set(AutozenFeatures.categorical_feat_list),
                            set(AutozenFeatures.ordinal_feat_list),
                            set(AutozenFeatures.numeric_feat_list),
                            set(AutozenFeatures.binary_feat_list),
                            set(AutozenFeatures.ordinal_feat_list_a),
                            set(AutozenFeatures.ordinal_feat_list_b),
                            set(AutozenFeatures.to_drop_feat_list)])

        print(set(AutozenFeatures.numeric_feat_list).intersection(set(AutozenFeatures.to_drop_feat_list)))
        # assert that they are the sum of the total
        remainder = subtract_subsets_from_superset(
            set(df_won.columns.tolist()),
            [set(AutozenFeatures.categorical_feat_list),
            set(AutozenFeatures.ordinal_feat_list),
            set(AutozenFeatures.numeric_feat_list),
            set(AutozenFeatures.binary_feat_list),
            set(AutozenFeatures.ordinal_feat_list_a),
            set(AutozenFeatures.ordinal_feat_list_b),
            set(AutozenFeatures.to_drop_feat_list)]
        )
        assert(len(remainder) == 1)
    
    
    @staticmethod
    def generate_preprocessor():
        return make_column_transformer(
            (AutozenFeatures.numeric_feat_transformer, AutozenFeatures.numeric_feat_list + ['car_age']),
            (AutozenFeatures.ordinal_feat_transformer, AutozenFeatures.ordinal_feat_list),
            (AutozenFeatures.categorical_feat_transformer, AutozenFeatures.categorical_feat_list),
            (AutozenFeatures.ordinal_feat_a_transformer, AutozenFeatures.ordinal_feat_list_a),
            (AutozenFeatures.ordinal_feat_b_transformer, AutozenFeatures.ordinal_feat_list_b),
            (AutozenFeatures.binary_feat_transformer, AutozenFeatures.binary_feat_list),
        )
    
    
    numeric_feat_transformer = make_pipeline(
         StandardScaler()
    )
    numeric_feat_list = [
        "general_mileage",
        "tires_driverRearPressure",
        "tires_driverFrontBreakPadMeasurement",
        "tires_passengerRearTreadDepth",
        "tires_passengerRearPressure",
        "tires_passengerFrontTreadDepth",
        "tires_driverFrontTreadDepth",
        "tires_passengerFrontBreakPadMeasurement",
        "tires_driverRearBreakPadMeasurement",
        "tires_driverFrontPressure",
        "tires_passengerFrontPressure",
        "tires_passengerRearBreakPadMeasurement",
        "tires_driverRearTreadDepth",
        "under_the_hood_coolantStrengthMeasurement",
        "under_the_hood_batteryRating",
        "under_the_hood_batteryActualMeasurement",
        "interior_numberOfKeysFobs",
        "interior_numberOfWorkingKeysFobs",
        "interior_numberOfKeysFobsDeclared",
        # "bid_winning",
        "bid_start",
        "car_age",
        # "num_auctions", # num_auctions
        "valuation_autozen_low",
        "valuation_autozen_high",
        "valuation_cbb_trade_in_low",
        "valuation_cbb_trade_in_high",
        "valuation_highest_bid_amount",
        "valuation_seller_asking_price",
        "valuation_starting_price"
    ]

    ordinal_feat_list = [
        "outside_driverPowerSlidingDoors",
        "outside_rustCorrosion",
        "outside_paintScratches",
        "outside_paintFinish",
        "outside_wiperBlades",
        "outside_collisionDamage",
        "outside_driverFrontWindowGlassOperation",
        "outside_driverFrontWindowGlass",
        "outside_passengerRearWindowGlass",
        "outside_bodyDentsDingsHailDamage",
        "outside_driverSideMirror",
        "outside_washerSprayNozzels",
        "outside_driverRearWindowGlassOperation",
        "outside_driverSideMirrorHeater",
        "outside_passengerRearWindowGlassOperation",
        "outside_passengerSideMirrorHeater",
        "outside_additionalIssuesExterior",
        "outside_liftgateOperation",
        "outside_passengerFrontWindowGlassOperation",
        "outside_driverRearWindowGlass",
        "outside_passengerPowerSlidingDoors",
        "outside_WindshieldGlass",
        "outside_passengerFrontWindowGlass",
        "outside_passengerSideMirror",
        "tires_driverRearShockStrut",
        "tires_passengerRearBrakePad",
        "tires_driverFrontCV",
        "tires_passengerFrontRotorCondition",
        "tires_passengerFrontBrakePad",
        "tires_driverFrontBrakePad",
        "tires_driverRearBrakePad",
        "tires_driverRearSideSwayBar",
        "tires_driverRearCV",
        "tires_additionalIssuesTires",
        "tires_driverRearCondition",
        "tires_driverRearRotorCondition",
        "tires_driverFrontTireCondition",
        "tires_passengerRearSideSwayBar",
        "tires_passengerFrontSideSwayBar",
        "tires_passengerRearTireCondition",
        "tires_driverFrontSideSwayBar",
        "tires_driverFrontCondition",
        "tires_driverRearValveStemCondition",
        "tires_passengerRearRotorCondition",
        "tires_passengerRearValveStemCondition",
        "tires_driverFrontRotorCondition",
        "tires_passengerRearCondition",
        "tires_driverFrontValveStemCondition",
        "tires_passengerFrontCondition",
        "tires_passengerRearCV",
        "tires_driverRearTireCondition",
        "tires_passengerFrontCV",
        "tires_passengerRearShockStrut",
        "tires_passengerFrontValveStemCondition",
        "tires_driverFrontShockStrut",
        "tires_passengerFrontTireCondition",
        "tires_passengerFrontShockStrut",
        "under_vehicle_frontDifferential",
        "under_vehicle_transferCase",
        "under_vehicle_additionalIssuesUnderVehicle",
        "under_vehicle_mufflerExhaustTips",
        "under_vehicle_rearDriveShaft",
        "under_vehicle_rearDifferential",
        "under_vehicle_frontDriveShaft",
        "under_the_hood_radiatorHosesClamps",
        "under_the_hood_engineOilCondition",
        "under_the_hood_coolantStrength",
        "under_the_hood_visibleOilLeaks",
        "under_the_hood_engineOilLevel",
        "under_the_hood_driveBelts",
        "under_the_hood_batteryStrength",
        "under_the_hood_additionalIssuesUnderHood",
        "under_the_hood_coolantOverflowTank",
        "under_the_hood_powerSteeringFluid",
        "under_the_hood_brakeFluid",
        "under_the_hood_starter",
        "under_the_hood_transmissionFluid",
        "under_the_hood_brakeMasterCylinder",
        "under_the_hood_acSystemLeaks",
        "under_the_hood_acCompressorOperation",
        "under_the_hood_coolingFanOperation",
        "under_the_hood_fuelSystemOperation",
        "under_the_hood_alternator",
        "under_the_hood_waterPumpNoiseLeaks",
        "lighting_headlightsHighBeam",
        "lighting_emergency4WayFlashers",
        "lighting_headlightsLowBeam",
        "lighting_brakeLights",
        "lighting_frontSignalLights",
        "lighting_interiorLights",
        "lighting_fogLights",
        "lighting_sideMarkerLights",
        "lighting_tailLights",
        "lighting_licensePlateLight",
        "lighting_rearSignalLights",
        "lighting_additionalIssuesLighting",
        "lighting_frontDRL",
        "lighting_reverseLights",
        "interior_conditionOfSeats",
        "interior_sunroofOperation",
        "interior_headrests",
        "interior_trunkRelease",
        "interior_seats",
        "interior_driverRearDoorTrim",
        "interior_steeringWheelSteeringColumn",
        "interior_evChargingKit",
        "interior_rearCamera",
        "interior_navigationOperation",
        "interior_rearWindowDefrostOperation",
        "interior_seatBelts",
        "interior_audioSystem",
        "interior_rearViewMirror",
        "interior_spareTireTools",
        "interior_gloveBoxCenterConsole",
        "interior_steeringWheel",
        "interior_spareTire",
        "interior_passengerFrontDoorTrim",
        "interior_cigaretteLighterPowerOutlets",
        "interior_hoodRelease",
        "interior_waterLeaks",
        "interior_keyFobCondition",
        "interior_passengerFrontSeatHeater",
        "interior_passengerRearDoorTrim",
        "interior_gloveBox",
        "interior_sunVisors",
        "interior_driverFrontDoorTrim",
        "interior_fuelDoorRelease",
        "interior_carpetFloorMats",
        "interior_driverRearSeatHeater",
        "interior_acOperationBlowsCold",
        "interior_headliner",
        "interior_audioSteeringWheelControls",
        "interior_horn",
        "interior_centerConsole",
        "interior_passengerRearSeatHeater",
        "interior_doorHandles",
        "interior_powerSeatOperation",
        "interior_dashboard",
        "interior_additionalIssuesInterior",
        "interior_driverFrontSeatHeater",
        "test_drive_cruiseControl",
        "test_drive_steeringWheelStraight",
        "test_drive_engineRPMIdle",
        "test_drive_alignmentCheck",
        "test_drive_additionalIssuesTestDrive",
        "test_drive_testDriveStatus",
        "test_drive_automaticManualTransmissionShift",
        "test_drive_parkingBreak",
        "test_drive_brakingSystem",
        "test_drive_cabinHeatBlowsHot",
        "test_drive_acOperationBlowsCold",
        "test_drive_engine3000RPM",
        "test_drive_abnormalWindNoise",
        "test_drive_clutchOperationEngagement",
        "test_drive_vibrationWhileDriving",
        "test_drive_vibrationWhileBraking",
        "test_drive_alignment",
    ]
    _repeated_list = [['notAvailable', 'serviceRequired', 'serviceSuggested', 'ok'] for _ in range(len(ordinal_feat_list))]
    ordinal_feat_transformer = make_pipeline(
          HandleUnknownTransformer(),
          OrdinalEncoder(categories=_repeated_list, handle_unknown='use_encoded_value', unknown_value=-1, dtype=int)
    )
    
    categorical_feat_transformer = make_pipeline(
        OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    )
    categorical_feat_list = ['general_make',
                            'general_model',
                            'general_trim',
                            'general_fuelType',
                            'general_roofType',
                            'general_transmission',
                            'general_warrantyProvider',
                            'vehicle_lead_source',
                            'seller_type',
                            'interior_odors',
                            'interior_seatsMaterial',
                            'general_year',
                            'month_of_year']

    ordinal_feat_list_a = ['general_additionalWarranty','general_transferableWarranty']
    _repeated_list_a = [['no', 'unknown', 'yes'] for _ in range(len(ordinal_feat_list_a))]
    ordinal_feat_a_transformer = make_pipeline(
          HandleUnknownTransformer(),
          OrdinalEncoder(categories=_repeated_list_a, handle_unknown='use_encoded_value', unknown_value=-1, dtype=int)
    )
    
    ordinal_feat_list_b = ["interior_spareTire", "interior_spareTireTools"]
    _repeated_list_b = [['no', 'notAvailable', 'serviceRequired', 'serviceSuggested', 'ok', 'yes'] for _ in range(len(ordinal_feat_list_b))]
    ordinal_feat_b_transformer = make_pipeline(
          HandleUnknownTransformer(),
          OrdinalEncoder(categories=_repeated_list_b, handle_unknown='use_encoded_value', unknown_value=-1, dtype=int)
    )
    
    binary_feat_transformer = make_pipeline(
        OneHotEncoder(drop="if_binary", dtype=int)
    )
    binary_feat_list = ['general_dashWarningLights',
                        'tires_aftermarketWheelsTires',
                        'tires_oemAvailable',
                        'tires_oemFactoryWheelsAvailable',
                        'tires_extraSetOfWheelsTiresAvailable',
                        'tires_secondSetOfTirePartOfDeal',
                        'test_drive_testDriveCompleted',
                        'interior_rearCameraEquipped',
                        'interior_sunroofEquipped',
                        'interior_navigationEquipped',
                        'interior_carSmokedIn']

    to_drop_feat_list = ['id_x','product_id',
                        'id_y','vehicle_vin',
                        'vehicle_mileage',
                        'id',
                        'Unnamed: 0',
                        'general_vinNumber', 
                        'status',
                        'num_auctions', # num_auctions
                        # 'created_at',
                        'test_drive_brakingSystemIncludingABS', # all nulls
                        'under_the_hood_batteryStrengthMeasurement', # all nulls
                        'tires_extraSetOfWheelsTires', # all nulls
                        'tires_aftermarketWheelsTiresHeader', # all nulls
                        'valuation_cbb_base_retail_avg',
                        'tires_passengervPressure', # all nulls
                        'interior_dashboardDriverSide', # all nulls
                        'interior_dashboardPassengerSide', # all nulls
                        'tires_confirmCustomerSecondWheelSet', # all nulls
                        'valuation_cbb_base_retail_avg', # all nulls
                        ]
