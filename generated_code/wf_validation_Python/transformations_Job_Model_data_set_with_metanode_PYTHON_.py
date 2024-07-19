import pandas as pd
import numpy as np
import functions.data_transformations as data_transformations
from helpers.enumerations import Belong, Operator, Operation, SpecialType, DataType, DerivedType, Closure, FilterType
from helpers.logger import set_logger

def generateWorkflow():
#-----------------New DataProcessing-----------------
#--------------------------------------Input data dictionaries--------------------------------------
	imputeMissingByMostFrequent_sex_IRISCHOOL_ETHNICITY__input_dataDictionary='./python_dataDictionaries/missing_input_dataDictionary.csv'
	imputeMissingByMostFrequent_sex_IRISCHOOL_ETHNICITY__input_dataDictionary_sep=','
	imputeMissingByFixValue_ACADEMIC_INTEREST_2_ACADEMIC_INTEREST_1__input_dataDictionary='./python_dataDictionaries/missing_input_dataDictionary.csv'
	imputeMissingByFixValue_ACADEMIC_INTEREST_2_ACADEMIC_INTEREST_1__input_dataDictionary_sep=','
	imputeMissingByMean_avg_income_distance__input_dataDictionary='./python_dataDictionaries/missing_input_dataDictionary.csv'
	imputeMissingByMean_avg_income_distance__input_dataDictionary_sep=','
	imputeMissingByLinearInterpolation_satscore__input_dataDictionary='./python_dataDictionaries/missing_input_dataDictionary.csv'
	imputeMissingByLinearInterpolation_satscore__input_dataDictionary_sep=','
	rowFilterRange_init_span__input_dataDictionary='./python_dataDictionaries/missing_output_dataDictionary.csv'
	rowFilterRange_init_span__input_dataDictionary_sep=','
	columnFilter_TRAVEL_INIT_CNTCTS_REFERRAL_CNCTS_telecq_interest_stuemail_CONTACT_CODE1__input_dataDictionary='./python_dataDictionaries/rowFilter_output_dataDictionary.csv'
	columnFilter_TRAVEL_INIT_CNTCTS_REFERRAL_CNCTS_telecq_interest_stuemail_CONTACT_CODE1__input_dataDictionary_sep=','
	mapping_TERRITORY__input_dataDictionary='./python_dataDictionaries/columnFilter_output_dataDictionary.csv'
	mapping_TERRITORY__input_dataDictionary_sep=','
	mapping_Instate__input_dataDictionary='./python_dataDictionaries/ruleEngine_territory_output_dataDictionary.csv'
	mapping_Instate__input_dataDictionary_sep=','
	stringToNumber_TERRITORY_Instate__input_dataDictionary='./python_dataDictionaries/ruleEngine_instate_output_dataDictionary.csv'
	stringToNumber_TERRITORY_Instate__input_dataDictionary_sep=','
	imputeOutlierByClosest_avg_income_distance_Instate__input_dataDictionary='./python_dataDictionaries/stringToNumber_output_dataDictionary.csv'
	imputeOutlierByClosest_avg_income_distance_Instate__input_dataDictionary_sep=','
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary='./python_dataDictionaries/numericOutliers_output_dataDictionary.csv'
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary_sep=','
	binner_TERRITORY__input_dataDictionary='./python_dataDictionaries/numericOutliers_output_dataDictionary.csv'
	binner_TERRITORY__input_dataDictionary_sep=','
	binner_satscore__input_dataDictionary='./python_dataDictionaries/numericOutliers_output_dataDictionary.csv'
	binner_satscore__input_dataDictionary_sep=','
	binner_avg_income__input_dataDictionary='./python_dataDictionaries/numericOutliers_output_dataDictionary.csv'
	binner_avg_income__input_dataDictionary_sep=','
#--------------------------------------Output data dictionaries--------------------------------------
	imputeMissingByMostFrequent_sex_IRISCHOOL_ETHNICITY__output_dataDictionary='./python_dataDictionaries/missing_output_dataDictionary.csv'
	imputeMissingByMostFrequent_sex_IRISCHOOL_ETHNICITY__output_dataDictionary_sep=','
	imputeMissingByFixValue_ACADEMIC_INTEREST_2_ACADEMIC_INTEREST_1__output_dataDictionary='./python_dataDictionaries/missing_output_dataDictionary.csv'
	imputeMissingByFixValue_ACADEMIC_INTEREST_2_ACADEMIC_INTEREST_1__output_dataDictionary_sep=','
	imputeMissingByMean_avg_income_distance__output_dataDictionary='./python_dataDictionaries/missing_output_dataDictionary.csv'
	imputeMissingByMean_avg_income_distance__output_dataDictionary_sep=','
	imputeMissingByLinearInterpolation_satscore__output_dataDictionary='./python_dataDictionaries/missing_output_dataDictionary.csv'
	imputeMissingByLinearInterpolation_satscore__output_dataDictionary_sep=','
	rowFilterRange_init_span__output_dataDictionary='./python_dataDictionaries/rowFilter_output_dataDictionary.csv'
	rowFilterRange_init_span__output_dataDictionary_sep=','
	columnFilter_TRAVEL_INIT_CNTCTS_REFERRAL_CNCTS_telecq_interest_stuemail_CONTACT_CODE1__output_dataDictionary='./python_dataDictionaries/columnFilter_output_dataDictionary.csv'
	columnFilter_TRAVEL_INIT_CNTCTS_REFERRAL_CNCTS_telecq_interest_stuemail_CONTACT_CODE1__output_dataDictionary_sep=','
	mapping_TERRITORY__output_dataDictionary='./python_dataDictionaries/ruleEngine_territory_output_dataDictionary.csv'
	mapping_TERRITORY__output_dataDictionary_sep=','
	mapping_Instate__output_dataDictionary='./python_dataDictionaries/ruleEngine_instate_output_dataDictionary.csv'
	mapping_Instate__output_dataDictionary_sep=','
	stringToNumber_TERRITORY_Instate__output_dataDictionary='./python_dataDictionaries/stringToNumber_output_dataDictionary.csv'
	stringToNumber_TERRITORY_Instate__output_dataDictionary_sep=','
	imputeOutlierByClosest_avg_income_distance_Instate__output_dataDictionary='./python_dataDictionaries/numericOutliers_output_dataDictionary.csv'
	imputeOutlierByClosest_avg_income_distance_Instate__output_dataDictionary_sep=','
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary='./python_dataDictionaries/numericBinner_output_dataDictionary.csv'
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary_sep=','
	binner_TERRITORY__output_dataDictionary='./python_dataDictionaries/numericBinner_output_dataDictionary.csv'
	binner_TERRITORY__output_dataDictionary_sep=','
	binner_satscore__output_dataDictionary='./python_dataDictionaries/numericBinner_output_dataDictionary.csv'
	binner_satscore__output_dataDictionary_sep=','
	binner_avg_income__output_dataDictionary='./python_dataDictionaries/numericBinner_output_dataDictionary.csv'
	binner_avg_income__output_dataDictionary_sep=','
#-----------------New DataProcessing-----------------
	imputeMissingByMostFrequent_sex_IRISCHOOL_ETHNICITY__input_dataDictionary=pd.read_csv(imputeMissingByMostFrequent_sex_IRISCHOOL_ETHNICITY__input_dataDictionary, sep = imputeMissingByMostFrequent_sex_IRISCHOOL_ETHNICITY__input_dataDictionary_sep)
	imputeMissingByMostFrequent_sex_IRISCHOOL_ETHNICITY__input_dataDictionary_transformed=imputeMissingByMostFrequent_sex_IRISCHOOL_ETHNICITY__input_dataDictionary.copy()
	missing_values_list=[]
	
	imputeMissingByMostFrequent_sex_IRISCHOOL_ETHNICITY__input_dataDictionary_transformed=data_transformations.transform_special_value_derived_value(data_dictionary=imputeMissingByMostFrequent_sex_IRISCHOOL_ETHNICITY__input_dataDictionary_transformed,
																  special_type_input=SpecialType(0), derived_type_output=DerivedType(0),
																  missing_values=missing_values_list,		
																  axis_param=0, field_in = 'sex', field_out = 'sex')
	
	missing_values_list=[]
	
	imputeMissingByMostFrequent_sex_IRISCHOOL_ETHNICITY__input_dataDictionary_transformed=data_transformations.transform_special_value_derived_value(data_dictionary=imputeMissingByMostFrequent_sex_IRISCHOOL_ETHNICITY__input_dataDictionary_transformed,
																  special_type_input=SpecialType(0), derived_type_output=DerivedType(0),
																  missing_values=missing_values_list,		
																  axis_param=0, field_in = 'IRSCHOOL', field_out = 'IRSCHOOL')
	
	missing_values_list=[]
	
	imputeMissingByMostFrequent_sex_IRISCHOOL_ETHNICITY__input_dataDictionary_transformed=data_transformations.transform_special_value_derived_value(data_dictionary=imputeMissingByMostFrequent_sex_IRISCHOOL_ETHNICITY__input_dataDictionary_transformed,
																  special_type_input=SpecialType(0), derived_type_output=DerivedType(0),
																  missing_values=missing_values_list,		
																  axis_param=0, field_in = 'ETHNICITY', field_out = 'ETHNICITY')
	
	imputeMissingByMostFrequent_sex_IRISCHOOL_ETHNICITY__output_dataDictionary_df=imputeMissingByMostFrequent_sex_IRISCHOOL_ETHNICITY__input_dataDictionary_transformed
	imputeMissingByMostFrequent_sex_IRISCHOOL_ETHNICITY__output_dataDictionary_df.to_csv(imputeMissingByMostFrequent_sex_IRISCHOOL_ETHNICITY__output_dataDictionary)	
	imputeMissingByMostFrequent_sex_IRISCHOOL_ETHNICITY__output_dataDictionary=pd.read_csv(imputeMissingByMostFrequent_sex_IRISCHOOL_ETHNICITY__output_dataDictionary, sep=imputeMissingByMostFrequent_sex_IRISCHOOL_ETHNICITY__output_dataDictionary_sep)
	
#-----------------New DataProcessing-----------------
	imputeMissingByFixValue_ACADEMIC_INTEREST_2_ACADEMIC_INTEREST_1__input_dataDictionary=pd.read_csv(imputeMissingByFixValue_ACADEMIC_INTEREST_2_ACADEMIC_INTEREST_1__input_dataDictionary, sep=imputeMissingByFixValue_ACADEMIC_INTEREST_2_ACADEMIC_INTEREST_1__input_dataDictionary_sep)

	imputeMissingByFixValue_ACADEMIC_INTEREST_2_ACADEMIC_INTEREST_1__input_dataDictionary_transformed=imputeMissingByFixValue_ACADEMIC_INTEREST_2_ACADEMIC_INTEREST_1__input_dataDictionary.copy()
	missing_values_list=[]
	
	imputeMissingByFixValue_ACADEMIC_INTEREST_2_ACADEMIC_INTEREST_1__input_dataDictionary_transformed=data_transformations.transform_special_value_fix_value(data_dictionary=imputeMissingByFixValue_ACADEMIC_INTEREST_2_ACADEMIC_INTEREST_1__input_dataDictionary_transformed,
																  special_type_input=SpecialType(0), fix_value_output='Unknown',
																  missing_values=missing_values_list,		
							                                      data_type_output = DataType(0),
																  axis_param=0, field_in = 'ACADEMIC_INTEREST_2', field_out = 'ACADEMIC_INTEREST_2')
	
	missing_values_list=[]
	
	imputeMissingByFixValue_ACADEMIC_INTEREST_2_ACADEMIC_INTEREST_1__input_dataDictionary_transformed=data_transformations.transform_special_value_fix_value(data_dictionary=imputeMissingByFixValue_ACADEMIC_INTEREST_2_ACADEMIC_INTEREST_1__input_dataDictionary_transformed,
																  special_type_input=SpecialType(0), fix_value_output='Unknown',
																  missing_values=missing_values_list,		
							                                      data_type_output = DataType(0),
																  axis_param=0, field_in = 'ACADEMIC_INTEREST_1', field_out = 'ACADEMIC_INTEREST_1')
	
	imputeMissingByFixValue_ACADEMIC_INTEREST_2_ACADEMIC_INTEREST_1__output_dataDictionary_df=imputeMissingByFixValue_ACADEMIC_INTEREST_2_ACADEMIC_INTEREST_1__input_dataDictionary_transformed
	imputeMissingByFixValue_ACADEMIC_INTEREST_2_ACADEMIC_INTEREST_1__output_dataDictionary_df.to_csv(imputeMissingByFixValue_ACADEMIC_INTEREST_2_ACADEMIC_INTEREST_1__output_dataDictionary)	
	imputeMissingByFixValue_ACADEMIC_INTEREST_2_ACADEMIC_INTEREST_1__output_dataDictionary=pd.read_csv(imputeMissingByFixValue_ACADEMIC_INTEREST_2_ACADEMIC_INTEREST_1__output_dataDictionary, sep=imputeMissingByFixValue_ACADEMIC_INTEREST_2_ACADEMIC_INTEREST_1__output_dataDictionary_sep)
	
#-----------------New DataProcessing-----------------
	imputeMissingByMean_avg_income_distance__input_dataDictionary=pd.read_csv(imputeMissingByMean_avg_income_distance__input_dataDictionary, sep=imputeMissingByMean_avg_income_distance__input_dataDictionary_sep)

	imputeMissingByMean_avg_income_distance__input_dataDictionary_transformed=imputeMissingByMean_avg_income_distance__input_dataDictionary.copy()
	missing_values_list=[]
	
	imputeMissingByMean_avg_income_distance__input_dataDictionary_transformed=data_transformations.transform_special_value_num_op(data_dictionary=imputeMissingByMean_avg_income_distance__input_dataDictionary_transformed,
																  special_type_input=SpecialType(0), num_op_output=Operation(1),
																  missing_values=missing_values_list,		
																  axis_param=0, field_in = 'avg_income', field_out = 'avg_income')
	
	missing_values_list=[]
	
	imputeMissingByMean_avg_income_distance__input_dataDictionary_transformed=data_transformations.transform_special_value_num_op(data_dictionary=imputeMissingByMean_avg_income_distance__input_dataDictionary_transformed,
																  special_type_input=SpecialType(0), num_op_output=Operation(1),
																  missing_values=missing_values_list,		
																  axis_param=0, field_in = 'distance', field_out = 'distance')
	
	imputeMissingByMean_avg_income_distance__output_dataDictionary_df=imputeMissingByMean_avg_income_distance__input_dataDictionary_transformed
	imputeMissingByMean_avg_income_distance__output_dataDictionary_df.to_csv(imputeMissingByMean_avg_income_distance__output_dataDictionary)	
	imputeMissingByMean_avg_income_distance__output_dataDictionary=pd.read_csv(imputeMissingByMean_avg_income_distance__output_dataDictionary, sep=imputeMissingByMean_avg_income_distance__output_dataDictionary_sep)
	
#-----------------New DataProcessing-----------------
	imputeMissingByLinearInterpolation_satscore__input_dataDictionary=pd.read_csv(imputeMissingByLinearInterpolation_satscore__input_dataDictionary, sep=imputeMissingByLinearInterpolation_satscore__input_dataDictionary_sep)

	imputeMissingByLinearInterpolation_satscore__input_dataDictionary_transformed=imputeMissingByLinearInterpolation_satscore__input_dataDictionary.copy()
	missing_values_list=[]
	
	imputeMissingByLinearInterpolation_satscore__input_dataDictionary_transformed=data_transformations.transform_special_value_num_op(data_dictionary=imputeMissingByLinearInterpolation_satscore__input_dataDictionary_transformed,
																  special_type_input=SpecialType(0), num_op_output=Operation(0),
																  missing_values=missing_values_list,		
																  axis_param=0, field_in = 'satscore', field_out = 'satscore')
	
	imputeMissingByLinearInterpolation_satscore__output_dataDictionary_df=imputeMissingByLinearInterpolation_satscore__input_dataDictionary_transformed
	imputeMissingByLinearInterpolation_satscore__output_dataDictionary_df.to_csv(imputeMissingByLinearInterpolation_satscore__output_dataDictionary)	
	imputeMissingByLinearInterpolation_satscore__output_dataDictionary=pd.read_csv(imputeMissingByLinearInterpolation_satscore__output_dataDictionary, sep=imputeMissingByLinearInterpolation_satscore__output_dataDictionary_sep)
	
#-----------------New DataProcessing-----------------
	rowFilterRange_init_span__input_dataDictionary=pd.read_csv(rowFilterRange_init_span__input_dataDictionary, sep=rowFilterRange_init_span__input_dataDictionary_sep)

	rowFilterRange_init_span__input_dataDictionary_transformed=rowFilterRange_init_span__input_dataDictionary.copy()
	columns_rowFilterRange_param_filter=['init_span']
	
	filter_range_left_values_list_rowFilterRange_param_filter=[-np.inf]
	filter_range_right_values_list_rowFilterRange_param_filter=[0.0]
	closure_type_list_rowFilterRange_param_filter=[Closure(3)]
	
	rowFilterRange_init_span__input_dataDictionary_transformed=data_transformations.transform_filter_rows_range(data_dictionary=rowFilterRange_init_span__input_dataDictionary_transformed,
																											columns=columns_rowFilterRange_param_filter,
																											left_margin_list=filter_range_left_values_list_rowFilterRange_param_filter,
																											right_margin_list=filter_range_right_values_list_rowFilterRange_param_filter,
																											filter_type=FilterType(0),
																											closure_type_list=closure_type_list_rowFilterRange_param_filter)
	rowFilterRange_init_span__output_dataDictionary_df=rowFilterRange_init_span__input_dataDictionary_transformed
	rowFilterRange_init_span__output_dataDictionary_df.to_csv(rowFilterRange_init_span__output_dataDictionary)	
	rowFilterRange_init_span__output_dataDictionary=pd.read_csv(rowFilterRange_init_span__output_dataDictionary, sep=rowFilterRange_init_span__output_dataDictionary_sep)
	
#-----------------New DataProcessing-----------------
	columnFilter_TRAVEL_INIT_CNTCTS_REFERRAL_CNCTS_telecq_interest_stuemail_CONTACT_CODE1__input_dataDictionary=pd.read_csv(columnFilter_TRAVEL_INIT_CNTCTS_REFERRAL_CNCTS_telecq_interest_stuemail_CONTACT_CODE1__input_dataDictionary, sep=columnFilter_TRAVEL_INIT_CNTCTS_REFERRAL_CNCTS_telecq_interest_stuemail_CONTACT_CODE1__input_dataDictionary_sep)

	columnFilter_TRAVEL_INIT_CNTCTS_REFERRAL_CNCTS_telecq_interest_stuemail_CONTACT_CODE1__input_dataDictionary_transformed=columnFilter_TRAVEL_INIT_CNTCTS_REFERRAL_CNCTS_telecq_interest_stuemail_CONTACT_CODE1__input_dataDictionary.copy()
	field_list_columnFilter_param_field=['TRAVEL_INIT_CNTCTS', 'REFERRAL_CNTCTS']
	
	columnFilter_TRAVEL_INIT_CNTCTS_REFERRAL_CNCTS_telecq_interest_stuemail_CONTACT_CODE1__input_dataDictionary_transformed=data_transformations.transform_filter_columns(data_dictionary=columnFilter_TRAVEL_INIT_CNTCTS_REFERRAL_CNCTS_telecq_interest_stuemail_CONTACT_CODE1__input_dataDictionary_transformed,
																	columns=field_list_columnFilter_param_field, belong_op=Belong.BELONG)
	
	columnFilter_TRAVEL_INIT_CNTCTS_REFERRAL_CNCTS_telecq_interest_stuemail_CONTACT_CODE1__output_dataDictionary_df=columnFilter_TRAVEL_INIT_CNTCTS_REFERRAL_CNCTS_telecq_interest_stuemail_CONTACT_CODE1__input_dataDictionary_transformed
	columnFilter_TRAVEL_INIT_CNTCTS_REFERRAL_CNCTS_telecq_interest_stuemail_CONTACT_CODE1__output_dataDictionary_df.to_csv(columnFilter_TRAVEL_INIT_CNTCTS_REFERRAL_CNCTS_telecq_interest_stuemail_CONTACT_CODE1__output_dataDictionary)	
	columnFilter_TRAVEL_INIT_CNTCTS_REFERRAL_CNCTS_telecq_interest_stuemail_CONTACT_CODE1__output_dataDictionary=pd.read_csv(columnFilter_TRAVEL_INIT_CNTCTS_REFERRAL_CNCTS_telecq_interest_stuemail_CONTACT_CODE1__output_dataDictionary, sep=columnFilter_TRAVEL_INIT_CNTCTS_REFERRAL_CNCTS_telecq_interest_stuemail_CONTACT_CODE1__output_dataDictionary_sep)
	
#-----------------New DataProcessing-----------------
	mapping_TERRITORY__input_dataDictionary=pd.read_csv(mapping_TERRITORY__input_dataDictionary, sep=mapping_TERRITORY__input_dataDictionary_sep)

	input_values_list=['A', 'N']
	output_values_list=['0', '0']
	data_type_input_list=[DataType(0), DataType(0)]
	data_type_output_list=[DataType(0), DataType(0)]
	
	
	mapping_TERRITORY__output_dataDictionary_df=data_transformations.transform_fix_value_fix_value(data_dictionary=mapping_TERRITORY__input_dataDictionary, input_values_list=input_values_list,
																  output_values_list=output_values_list,
							                                      data_type_input_list = data_type_input_list,
							                                      data_type_output_list = data_type_output_list, field_in = 'TERRITORY', field_out = 'TERRITORY')
	
	mapping_TERRITORY__output_dataDictionary_df.to_csv(mapping_TERRITORY__output_dataDictionary)	
	mapping_TERRITORY__output_dataDictionary=pd.read_csv(mapping_TERRITORY__output_dataDictionary, sep=mapping_TERRITORY__output_dataDictionary_sep)
	
#-----------------New DataProcessing-----------------
	mapping_Instate__input_dataDictionary=pd.read_csv(mapping_Instate__input_dataDictionary, sep=mapping_Instate__input_dataDictionary_sep)

	input_values_list=['Y', 'N']
	output_values_list=['1', '0']
	data_type_input_list=[DataType(0), DataType(0)]
	data_type_output_list=[DataType(0), DataType(0)]
	
	
	mapping_Instate__output_dataDictionary_df=data_transformations.transform_fix_value_fix_value(data_dictionary=mapping_Instate__input_dataDictionary, input_values_list=input_values_list,
																  output_values_list=output_values_list,
							                                      data_type_input_list = data_type_input_list,
							                                      data_type_output_list = data_type_output_list, field_in = 'Instate', field_out = 'Instate')
	
	mapping_Instate__output_dataDictionary_df.to_csv(mapping_Instate__output_dataDictionary)	
	mapping_Instate__output_dataDictionary=pd.read_csv(mapping_Instate__output_dataDictionary, sep=mapping_Instate__output_dataDictionary_sep)
	
#-----------------New DataProcessing-----------------
	stringToNumber_TERRITORY_Instate__input_dataDictionary=pd.read_csv(stringToNumber_TERRITORY_Instate__input_dataDictionary, sep=stringToNumber_TERRITORY_Instate__input_dataDictionary_sep)

	stringToNumber_TERRITORY_Instate__input_dataDictionary_transformed=stringToNumber_TERRITORY_Instate__input_dataDictionary.copy()
	stringToNumber_TERRITORY_Instate__input_dataDictionary_transformed=data_transformations.transform_cast_type(data_dictionary=stringToNumber_TERRITORY_Instate__input_dataDictionary_transformed,
																	data_type_output= DataType(6),
																	field='TERRITORY')
	
	stringToNumber_TERRITORY_Instate__input_dataDictionary_transformed=data_transformations.transform_cast_type(data_dictionary=stringToNumber_TERRITORY_Instate__input_dataDictionary_transformed,
																	data_type_output= DataType(6),
																	field='Instate')
	
	stringToNumber_TERRITORY_Instate__output_dataDictionary_df=stringToNumber_TERRITORY_Instate__input_dataDictionary_transformed
	stringToNumber_TERRITORY_Instate__output_dataDictionary_df.to_csv(stringToNumber_TERRITORY_Instate__output_dataDictionary)	
	stringToNumber_TERRITORY_Instate__output_dataDictionary=pd.read_csv(stringToNumber_TERRITORY_Instate__output_dataDictionary, sep=stringToNumber_TERRITORY_Instate__output_dataDictionary_sep)
	
#-----------------New DataProcessing-----------------
	imputeOutlierByClosest_avg_income_distance_Instate__input_dataDictionary=pd.read_csv(imputeOutlierByClosest_avg_income_distance_Instate__input_dataDictionary, sep=imputeOutlierByClosest_avg_income_distance_Instate__input_dataDictionary_sep)

	imputeOutlierByClosest_avg_income_distance_Instate__input_dataDictionary_transformed=imputeOutlierByClosest_avg_income_distance_Instate__input_dataDictionary.copy()
	missing_values_list=[]
	
	imputeOutlierByClosest_avg_income_distance_Instate__input_dataDictionary_transformed=data_transformations.transform_special_value_num_op(data_dictionary=imputeOutlierByClosest_avg_income_distance_Instate__input_dataDictionary_transformed,
																  special_type_input=SpecialType(2), num_op_output=Operation(3),
																  missing_values=missing_values_list,		
																  axis_param=0, field_in = 'avg_income', field_out = 'avg_income')
	
	missing_values_list=[]
	
	imputeOutlierByClosest_avg_income_distance_Instate__input_dataDictionary_transformed=data_transformations.transform_special_value_num_op(data_dictionary=imputeOutlierByClosest_avg_income_distance_Instate__input_dataDictionary_transformed,
																  special_type_input=SpecialType(2), num_op_output=Operation(3),
																  missing_values=missing_values_list,		
																  axis_param=0, field_in = 'distance', field_out = 'distance')
	
	missing_values_list=[]
	
	imputeOutlierByClosest_avg_income_distance_Instate__input_dataDictionary_transformed=data_transformations.transform_special_value_num_op(data_dictionary=imputeOutlierByClosest_avg_income_distance_Instate__input_dataDictionary_transformed,
																  special_type_input=SpecialType(2), num_op_output=Operation(3),
																  missing_values=missing_values_list,		
																  axis_param=0, field_in = 'Instate', field_out = 'Instate')
	
	imputeOutlierByClosest_avg_income_distance_Instate__output_dataDictionary_df=imputeOutlierByClosest_avg_income_distance_Instate__input_dataDictionary_transformed
	imputeOutlierByClosest_avg_income_distance_Instate__output_dataDictionary_df.to_csv(imputeOutlierByClosest_avg_income_distance_Instate__output_dataDictionary)	
	imputeOutlierByClosest_avg_income_distance_Instate__output_dataDictionary=pd.read_csv(imputeOutlierByClosest_avg_income_distance_Instate__output_dataDictionary, sep=imputeOutlierByClosest_avg_income_distance_Instate__output_dataDictionary_sep)
	
#-----------------New DataProcessing-----------------
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary=pd.read_csv(binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary, sep=binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary_sep)

	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary_transformed=binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary.copy()
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary_transformed=data_transformations.transform_derived_field(data_dictionary=binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary_transformed,
																  data_type_output = DataType(0),
																  field_in = 'TOTAL_CONTACTS', field_out = 'TOTAL_CONTACTS_binned')
	
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary_df=binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary_transformed
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary_df.to_csv(binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary)	
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary=pd.read_csv(binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary, sep=binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary_sep)
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary_transformed=data_transformations.transform_derived_field(data_dictionary=binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary_transformed,
																  data_type_output = DataType(0),
																  field_in = 'SELF_INIT_CNTCTS', field_out = 'SELF_INIT_CNTCTS_binned')
	
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary_df=binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary_transformed
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary_df.to_csv(binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary)	
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary=pd.read_csv(binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary, sep=binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary_sep)
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary_transformed=data_transformations.transform_derived_field(data_dictionary=binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary_transformed,
																  data_type_output = DataType(0),
																  field_in = 'SOLICITED_CNTCTS', field_out = 'SOLICITED_CNTCTS_binned')
	
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary_df=binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary_transformed
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary_df.to_csv(binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary)	
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary=pd.read_csv(binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary, sep=binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary_sep)
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary_transformed,
																  left_margin=-1000.0, right_margin=1.0,
																  closure_type=Closure(0),
																  fix_value_output='Low',
							                                      data_type_output = DataType(0),
																  field_in = 'TOTAL_CONTACTS',
																  field_out = 'TOTAL_CONTACTS_binned')
	
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary_transformed,
																  left_margin=-1000.0, right_margin=1.0,
																  closure_type=Closure(0),
																  fix_value_output='Low',
							                                      data_type_output = DataType(0),
																  field_in = 'SELF_INIT_CNTCTS',
																  field_out = 'SELF_INIT_CNTCTS_binned')
	
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary_transformed,
																  left_margin=-1000.0, right_margin=1.0,
																  closure_type=Closure(0),
																  fix_value_output='Low',
							                                      data_type_output = DataType(0),
																  field_in = 'SOLICITED_CNTCTS',
																  field_out = 'SOLICITED_CNTCTS_binned')
	
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary_df=binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary_transformed
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary_df.to_csv(binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary)	
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary=pd.read_csv(binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary, sep=binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary_sep)
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary_transformed,
																  left_margin=1.0, right_margin=4.0,
																  closure_type=Closure(2),
																  fix_value_output='Moderate',
							                                      data_type_output = DataType(0),
																  field_in = 'TOTAL_CONTACTS',
																  field_out = 'TOTAL_CONTACTS_binned')
	
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary_transformed,
																  left_margin=1.0, right_margin=4.0,
																  closure_type=Closure(2),
																  fix_value_output='Moderate',
							                                      data_type_output = DataType(0),
																  field_in = 'SELF_INIT_CNTCTS',
																  field_out = 'SELF_INIT_CNTCTS_binned')
	
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary_transformed,
																  left_margin=1.0, right_margin=4.0,
																  closure_type=Closure(2),
																  fix_value_output='Moderate',
							                                      data_type_output = DataType(0),
																  field_in = 'SOLICITED_CNTCTS',
																  field_out = 'SOLICITED_CNTCTS_binned')
	
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary_df=binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary_transformed
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary_df.to_csv(binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary)	
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary=pd.read_csv(binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary, sep=binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary_sep)
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary_transformed,
																  left_margin=4.0, right_margin=1000.0,
																  closure_type=Closure(2),
																  fix_value_output='High',
							                                      data_type_output = DataType(0),
																  field_in = 'TOTAL_CONTACTS',
																  field_out = 'TOTAL_CONTACTS_binned')
	
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary_transformed,
																  left_margin=4.0, right_margin=1000.0,
																  closure_type=Closure(2),
																  fix_value_output='High',
							                                      data_type_output = DataType(0),
																  field_in = 'SELF_INIT_CNTCTS',
																  field_out = 'SELF_INIT_CNTCTS_binned')
	
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary_transformed,
																  left_margin=4.0, right_margin=1000.0,
																  closure_type=Closure(2),
																  fix_value_output='High',
							                                      data_type_output = DataType(0),
																  field_in = 'SOLICITED_CNTCTS',
																  field_out = 'SOLICITED_CNTCTS_binned')
	
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary_df=binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__input_dataDictionary_transformed
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary_df.to_csv(binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary)	
	binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary=pd.read_csv(binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary, sep=binner_TOTAL_CONTACTS_SELF_INIT_CNTCTS_SOLICITED_CNTCTS__output_dataDictionary_sep)
	
#-----------------New DataProcessing-----------------
	binner_TERRITORY__input_dataDictionary=pd.read_csv(binner_TERRITORY__input_dataDictionary, sep=binner_TERRITORY__input_dataDictionary_sep)

	binner_TERRITORY__input_dataDictionary_transformed=binner_TERRITORY__input_dataDictionary.copy()
	binner_TERRITORY__input_dataDictionary_transformed=data_transformations.transform_derived_field(data_dictionary=binner_TERRITORY__input_dataDictionary_transformed,
																  data_type_output = DataType(0),
																  field_in = 'TERRITORY', field_out = 'TERRITORY_binned')
	
	binner_TERRITORY__output_dataDictionary_df=binner_TERRITORY__input_dataDictionary_transformed
	binner_TERRITORY__output_dataDictionary_df.to_csv(binner_TERRITORY__output_dataDictionary)	
	binner_TERRITORY__output_dataDictionary=pd.read_csv(binner_TERRITORY__output_dataDictionary, sep=binner_TERRITORY__output_dataDictionary_sep)
	binner_TERRITORY__input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_TERRITORY__input_dataDictionary_transformed,
																  left_margin=-1000.0, right_margin=1.0,
																  closure_type=Closure(0),
																  fix_value_output='Unknown',
							                                      data_type_output = DataType(0),
																  field_in = 'TERRITORY',
																  field_out = 'TERRITORY_binned')
	
	binner_TERRITORY__output_dataDictionary_df=binner_TERRITORY__input_dataDictionary_transformed
	binner_TERRITORY__output_dataDictionary_df.to_csv(binner_TERRITORY__output_dataDictionary)	
	binner_TERRITORY__output_dataDictionary=pd.read_csv(binner_TERRITORY__output_dataDictionary, sep=binner_TERRITORY__output_dataDictionary_sep)
	binner_TERRITORY__input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_TERRITORY__input_dataDictionary_transformed,
																  left_margin=1.0, right_margin=3.0,
																  closure_type=Closure(2),
																  fix_value_output='Zone 1',
							                                      data_type_output = DataType(0),
																  field_in = 'TERRITORY',
																  field_out = 'TERRITORY_binned')
	
	binner_TERRITORY__output_dataDictionary_df=binner_TERRITORY__input_dataDictionary_transformed
	binner_TERRITORY__output_dataDictionary_df.to_csv(binner_TERRITORY__output_dataDictionary)	
	binner_TERRITORY__output_dataDictionary=pd.read_csv(binner_TERRITORY__output_dataDictionary, sep=binner_TERRITORY__output_dataDictionary_sep)
	binner_TERRITORY__input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_TERRITORY__input_dataDictionary_transformed,
																  left_margin=3.0, right_margin=5.0,
																  closure_type=Closure(2),
																  fix_value_output='Zone 2',
							                                      data_type_output = DataType(0),
																  field_in = 'TERRITORY',
																  field_out = 'TERRITORY_binned')
	
	binner_TERRITORY__output_dataDictionary_df=binner_TERRITORY__input_dataDictionary_transformed
	binner_TERRITORY__output_dataDictionary_df.to_csv(binner_TERRITORY__output_dataDictionary)	
	binner_TERRITORY__output_dataDictionary=pd.read_csv(binner_TERRITORY__output_dataDictionary, sep=binner_TERRITORY__output_dataDictionary_sep)
	binner_TERRITORY__input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_TERRITORY__input_dataDictionary_transformed,
																  left_margin=5.0, right_margin=7.0,
																  closure_type=Closure(2),
																  fix_value_output='Zone 3',
							                                      data_type_output = DataType(0),
																  field_in = 'TERRITORY',
																  field_out = 'TERRITORY_binned')
	
	binner_TERRITORY__output_dataDictionary_df=binner_TERRITORY__input_dataDictionary_transformed
	binner_TERRITORY__output_dataDictionary_df.to_csv(binner_TERRITORY__output_dataDictionary)	
	binner_TERRITORY__output_dataDictionary=pd.read_csv(binner_TERRITORY__output_dataDictionary, sep=binner_TERRITORY__output_dataDictionary_sep)
	binner_TERRITORY__input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_TERRITORY__input_dataDictionary_transformed,
																  left_margin=7.0, right_margin=1000.0,
																  closure_type=Closure(3),
																  fix_value_output='Zone 4',
							                                      data_type_output = DataType(0),
																  field_in = 'TERRITORY',
																  field_out = 'TERRITORY_binned')
	
	binner_TERRITORY__output_dataDictionary_df=binner_TERRITORY__input_dataDictionary_transformed
	binner_TERRITORY__output_dataDictionary_df.to_csv(binner_TERRITORY__output_dataDictionary)	
	binner_TERRITORY__output_dataDictionary=pd.read_csv(binner_TERRITORY__output_dataDictionary, sep=binner_TERRITORY__output_dataDictionary_sep)
	
#-----------------New DataProcessing-----------------
	binner_satscore__input_dataDictionary=pd.read_csv(binner_satscore__input_dataDictionary, sep=binner_satscore__input_dataDictionary_sep)

	binner_satscore__input_dataDictionary_transformed=binner_satscore__input_dataDictionary.copy()
	binner_satscore__input_dataDictionary_transformed=data_transformations.transform_derived_field(data_dictionary=binner_satscore__input_dataDictionary_transformed,
																  data_type_output = DataType(0),
																  field_in = 'satscore', field_out = 'satscore_binned')
	
	binner_satscore__output_dataDictionary_df=binner_satscore__input_dataDictionary_transformed
	binner_satscore__output_dataDictionary_df.to_csv(binner_satscore__output_dataDictionary)	
	binner_satscore__output_dataDictionary=pd.read_csv(binner_satscore__output_dataDictionary, sep=binner_satscore__output_dataDictionary_sep)
	binner_satscore__input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_satscore__input_dataDictionary_transformed,
																  left_margin=-1000.0, right_margin=1040.0,
																  closure_type=Closure(1),
																  fix_value_output='54 Percentile and Under',
							                                      data_type_output = DataType(0),
																  field_in = 'satscore',
																  field_out = 'satscore_binned')
	
	binner_satscore__output_dataDictionary_df=binner_satscore__input_dataDictionary_transformed
	binner_satscore__output_dataDictionary_df.to_csv(binner_satscore__output_dataDictionary)	
	binner_satscore__output_dataDictionary=pd.read_csv(binner_satscore__output_dataDictionary, sep=binner_satscore__output_dataDictionary_sep)
	binner_satscore__input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_satscore__input_dataDictionary_transformed,
																  left_margin=1040.0, right_margin=1160.0,
																  closure_type=Closure(0),
																  fix_value_output='55-75 Percentile',
							                                      data_type_output = DataType(0),
																  field_in = 'satscore',
																  field_out = 'satscore_binned')
	
	binner_satscore__output_dataDictionary_df=binner_satscore__input_dataDictionary_transformed
	binner_satscore__output_dataDictionary_df.to_csv(binner_satscore__output_dataDictionary)	
	binner_satscore__output_dataDictionary=pd.read_csv(binner_satscore__output_dataDictionary, sep=binner_satscore__output_dataDictionary_sep)
	binner_satscore__input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_satscore__input_dataDictionary_transformed,
																  left_margin=1160.0, right_margin=1340.0,
																  closure_type=Closure(2),
																  fix_value_output='76-93 Percentile',
							                                      data_type_output = DataType(0),
																  field_in = 'satscore',
																  field_out = 'satscore_binned')
	
	binner_satscore__output_dataDictionary_df=binner_satscore__input_dataDictionary_transformed
	binner_satscore__output_dataDictionary_df.to_csv(binner_satscore__output_dataDictionary)	
	binner_satscore__output_dataDictionary=pd.read_csv(binner_satscore__output_dataDictionary, sep=binner_satscore__output_dataDictionary_sep)
	binner_satscore__input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_satscore__input_dataDictionary_transformed,
																  left_margin=1340.0, right_margin=2000.0,
																  closure_type=Closure(1),
																  fix_value_output='94+ percentile',
							                                      data_type_output = DataType(0),
																  field_in = 'satscore',
																  field_out = 'satscore_binned')
	
	binner_satscore__output_dataDictionary_df=binner_satscore__input_dataDictionary_transformed
	binner_satscore__output_dataDictionary_df.to_csv(binner_satscore__output_dataDictionary)	
	binner_satscore__output_dataDictionary=pd.read_csv(binner_satscore__output_dataDictionary, sep=binner_satscore__output_dataDictionary_sep)
	
#-----------------New DataProcessing-----------------
	binner_avg_income__input_dataDictionary=pd.read_csv(binner_avg_income__input_dataDictionary, sep=binner_avg_income__input_dataDictionary_sep)

	binner_avg_income__input_dataDictionary_transformed=binner_avg_income__input_dataDictionary.copy()
	binner_avg_income__input_dataDictionary_transformed=data_transformations.transform_derived_field(data_dictionary=binner_avg_income__input_dataDictionary_transformed,
																  data_type_output = DataType(0),
																  field_in = 'avg_income', field_out = 'avg_income_binned')
	
	binner_avg_income__output_dataDictionary_df=binner_avg_income__input_dataDictionary_transformed
	binner_avg_income__output_dataDictionary_df.to_csv(binner_avg_income__output_dataDictionary)	
	binner_avg_income__output_dataDictionary=pd.read_csv(binner_avg_income__output_dataDictionary, sep=binner_avg_income__output_dataDictionary_sep)
	binner_avg_income__input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_avg_income__input_dataDictionary_transformed,
																  left_margin=9.0, right_margin=42830.0,
																  closure_type=Closure(1),
																  fix_value_output='Low',
							                                      data_type_output = DataType(0),
																  field_in = 'avg_income',
																  field_out = 'avg_income_binned')
	
	binner_avg_income__output_dataDictionary_df=binner_avg_income__input_dataDictionary_transformed
	binner_avg_income__output_dataDictionary_df.to_csv(binner_avg_income__output_dataDictionary)	
	binner_avg_income__output_dataDictionary=pd.read_csv(binner_avg_income__output_dataDictionary, sep=binner_avg_income__output_dataDictionary_sep)
	binner_avg_income__input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_avg_income__input_dataDictionary_transformed,
																  left_margin=42830.0, right_margin=55590.0,
																  closure_type=Closure(1),
																  fix_value_output='Moderate',
							                                      data_type_output = DataType(0),
																  field_in = 'avg_income',
																  field_out = 'avg_income_binned')
	
	binner_avg_income__output_dataDictionary_df=binner_avg_income__input_dataDictionary_transformed
	binner_avg_income__output_dataDictionary_df.to_csv(binner_avg_income__output_dataDictionary)	
	binner_avg_income__output_dataDictionary=pd.read_csv(binner_avg_income__output_dataDictionary, sep=binner_avg_income__output_dataDictionary_sep)
	binner_avg_income__input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_avg_income__input_dataDictionary_transformed,
																  left_margin=55590.0, right_margin=100000.0,
																  closure_type=Closure(2),
																  fix_value_output='High',
							                                      data_type_output = DataType(0),
																  field_in = 'avg_income',
																  field_out = 'avg_income_binned')
	
	binner_avg_income__output_dataDictionary_df=binner_avg_income__input_dataDictionary_transformed
	binner_avg_income__output_dataDictionary_df.to_csv(binner_avg_income__output_dataDictionary)	
	binner_avg_income__output_dataDictionary=pd.read_csv(binner_avg_income__output_dataDictionary, sep=binner_avg_income__output_dataDictionary_sep)
	














set_logger("transformations")
generateWorkflow()
