
import pandas as pd
import functions.data_transformations as data_transformations
from helpers.enumerations import Belong, Operator, Operation, SpecialType, DataType, DerivedType, Closure
class DataProcessing:
	def generateDataProcessing(self):
		transformations=data_transformations.DataTransformations()
#-----------------New DataProcessing-----------------
		data_model_impute_in=pd.read_csv('./data_model.csv')
		data_model_impute_in_transformed=data_model_impute_in.copy()
		missing_values_list_parameter_derivedValue_impute_mostFrequent=['4']
		data_model_impute_in_transformed=transformations.transform_special_value_derived_value(data_dictionary=data_model_impute_in_transformed,
																	  special_type_input=SpecialType(0), derived_type_output=DerivedType(0),
																	  missing_values=missing_values_list_parameter_derivedValue_impute_mostFrequent,
																	  axis_param=0, field = 'sex')
		missing_values_list_parameter_derivedValue_impute_mostFrequent=[]
		data_model_impute_in_transformed=transformations.transform_special_value_derived_value(data_dictionary=data_model_impute_in_transformed,
																	  special_type_input=SpecialType(0), derived_type_output=DerivedType(0),
																	  missing_values=missing_values_list_parameter_derivedValue_impute_mostFrequent,
																	  axis_param=0, field = 'IRSCHOOL')
		missing_values_list_parameter_derivedValue_impute_mostFrequent=[]
		data_model_impute_in_transformed=transformations.transform_special_value_derived_value(data_dictionary=data_model_impute_in_transformed,
																	  special_type_input=SpecialType(0), derived_type_output=DerivedType(0),
																	  missing_values=missing_values_list_parameter_derivedValue_impute_mostFrequent,
																	  axis_param=0, field = 'ETHNICITY')
		
		data_model_impute_out=data_model_impute_in_transformed
		data_model_impute_out.to_csv('./workflow_datasets/data_model_impute_out.csv')
		
#-----------------New DataProcessing-----------------
		data_model_impute_ACADEMIC_INTEREST_2_in=pd.read_csv('./workflow_datasets/data_model_impute_out.csv')
		data_model_impute_ACADEMIC_INTEREST_2_in_transformed=data_model_impute_ACADEMIC_INTEREST_2_in.copy()
		missing_values_list_parameter_fixValue_impute_fixValue=[]
		data_model_impute_ACADEMIC_INTEREST_2_in_transformed=transformations.transform_special_value_fix_value(data_dictionary=data_model_impute_ACADEMIC_INTEREST_2_in_transformed,
																	  special_type_input=SpecialType(0), fix_value_output='Unknown',
																	  missing_values=missing_values_list_parameter_fixValue_impute_fixValue,
								                                      data_type_output = DataType(0),
																	  axis_param=0, field = 'ACADEMIC_INTEREST_2')
		missing_values_list_parameter_fixValue_impute_fixValue=[]
		data_model_impute_ACADEMIC_INTEREST_2_in_transformed=transformations.transform_special_value_fix_value(data_dictionary=data_model_impute_ACADEMIC_INTEREST_2_in_transformed,
																	  special_type_input=SpecialType(0), fix_value_output='Unknown',
																	  missing_values=missing_values_list_parameter_fixValue_impute_fixValue,
								                                      data_type_output = DataType(0),
																	  axis_param=0, field = 'ACADEMIC_INTEREST_1')
		
		data_model_impute_ACADEMIC_INTEREST_2_out=data_model_impute_ACADEMIC_INTEREST_2_in_transformed
		data_model_impute_ACADEMIC_INTEREST_2_out.to_csv('./workflow_datasets/data_model_impute_out2.csv')
		
#-----------------New DataProcessing-----------------
# 		data_model_impute_mean_in=pd.read_csv('./workflow_datasets/data_model_impute_out2.csv')
# 		data_model_impute_mean_in_transformed=data_model_impute_mean_in.copy()
# 		missing_values_list_parameter_num_op_impute_mean=[]
# 		data_model_impute_mean_in_transformed=transformations.transform_special_value_num_op(data_dictionary=data_model_impute_mean_in_transformed,
# 																	  special_type_input=SpecialType(0), num_op_output=Operation(1),
# 																	  missing_values=missing_values_list_parameter_num_op_impute_mean,
# 																	  axis_param=0, field = 'avg_income')
# 		missing_values_list_parameter_num_op_impute_mean=[]
# 		data_model_impute_mean_in_transformed=transformations.transform_special_value_num_op(data_dictionary=data_model_impute_mean_in_transformed,
# 																	  special_type_input=SpecialType(0), num_op_output=Operation(1),
# 																	  missing_values=missing_values_list_parameter_num_op_impute_mean,
# 																	  axis_param=0, field = 'distance')
#
# 		data_model_impute_mean_out=data_model_impute_mean_in_transformed
# 		data_model_impute_mean_out.to_csv('./workflow_datasets/data_model_impute_out3.csv')
#
# #-----------------New DataProcessing-----------------
# 		data_model_impute_linear_interpolation_in=pd.read_csv('./workflow_datasets/data_model_impute_out3.csv')
# 		data_model_impute_linear_interpolation_in_transformed=data_model_impute_linear_interpolation_in.copy()
# 		missing_values_list_parameter_num_op_impute_mean=[]
# 		data_model_impute_linear_interpolation_in_transformed=transformations.transform_special_value_num_op(data_dictionary=data_model_impute_linear_interpolation_in_transformed,
# 																	  special_type_input=SpecialType(0), num_op_output=Operation(1),
# 																	  missing_values=missing_values_list_parameter_num_op_impute_mean,
# 																	  axis_param=0, field = 'satscore')
#
# 		data_model_impute_linear_interpolation_out=data_model_impute_linear_interpolation_in_transformed
# 		data_model_impute_linear_interpolation_out.to_csv('./workflow_datasets/data_model_impute_out4.csv')
#
# #-----------------New DataProcessing-----------------
# 		data_model_row_filter_in=pd.read_csv('./workflow_datasets/data_model_impute_out4.csv')
# 		data_model_row_filter_in_transformed=data_model_row_filter_in.copy()
#
# #-----------------New DataProcessing-----------------
# 		data_model_column_cont_filter_in=pd.read_csv('./workflow_datasets/data_model_row_filter_out.csv')
# 		data_model_column_cont_filter_in_transformed=data_model_column_cont_filter_in.copy()
#
# #-----------------New DataProcessing-----------------
# 		data_model_column_cat_filter_in=pd.read_csv('./workflow_datasets/data_model_col_filter_out.csv')
# 		data_model_column_cat_filter_in_transformed=data_model_column_cat_filter_in.copy()
#
# #-----------------New DataProcessing-----------------
# 		data_model_map_territory_in=pd.read_csv('./workflow_datasets/data_model_col_filter_out2.csv')
# 		input_values_list_parameter_map_territory_A=['A', 'N']
# 		output_values_list_parameter_map_territory_A=['0', '0']
# 		data_type_input_list_parameter_map_territory_A=[DataType(0), DataType(0)]
# 		data_type_output_list_parameter_map_territory_A=[DataType(0), DataType(0)]
#
#
# 		data_model_map_territory_out=transformations.transform_fix_value_fix_value(data_dictionary=data_model_map_territory_in, input_values_list=input_values_list_parameter_map_territory_A,
# 																	  output_values_list=output_values_list_parameter_map_territory_A,
# 								                                      data_type_input_list = data_type_input_list_parameter_map_territory_A,
# 								                                      data_type_output_list = data_type_output_list_parameter_map_territory_A, field = 'TERRITORY')
#
# 		data_model_map_territory_out.to_csv('./workflow_datasets/data_model_map_territory_out.csv')
#
# #-----------------New DataProcessing-----------------
# 		data_model_map_Instate_in=pd.read_csv('./workflow_datasets/data_model_map_territory_out.csv')
# 		input_values_list_parameter_map_instate_Y=['Y', 'N']
# 		output_values_list_parameter_map_instate_Y=['1', '0']
# 		data_type_input_list_parameter_map_instate_Y=[DataType(0), DataType(0)]
# 		data_type_output_list_parameter_map_instate_Y=[DataType(0), DataType(0)]
#
#
# 		data_model_map_Instate_out=transformations.transform_fix_value_fix_value(data_dictionary=data_model_map_Instate_in, input_values_list=input_values_list_parameter_map_instate_Y,
# 																	  output_values_list=output_values_list_parameter_map_instate_Y,
# 								                                      data_type_input_list = data_type_input_list_parameter_map_instate_Y,
# 								                                      data_type_output_list = data_type_output_list_parameter_map_instate_Y, field = 'Instate')
#
# 		data_model_map_Instate_out.to_csv('./workflow_datasets/data_model_map_instate_out.csv')
#
# #-----------------New DataProcessing-----------------
# 		data_model_stringToNumber_in=pd.read_csv('./workflow_datasets/data_model_map_instate_out.csv')
#
# #-----------------New DataProcessing-----------------
# 		data_model_impute_outlier_closest_in=pd.read_csv('./workflow_datasets/data_model_stringToNumber_out.csv')
# 		data_model_impute_outlier_closest_in_transformed=data_model_impute_outlier_closest_in.copy()
# 		missing_values_list_parameter_num_op_impute_mean=[]
# 		data_model_impute_outlier_closest_in_transformed=transformations.transform_special_value_num_op(data_dictionary=data_model_impute_outlier_closest_in_transformed,
# 																	  special_type_input=SpecialType(2), num_op_output=Operation(3),
# 																	  missing_values=missing_values_list_parameter_num_op_impute_mean,
# 																	  axis_param=0, field = 'avg_income')
# 		missing_values_list_parameter_num_op_impute_mean=[]
# 		data_model_impute_outlier_closest_in_transformed=transformations.transform_special_value_num_op(data_dictionary=data_model_impute_outlier_closest_in_transformed,
# 																	  special_type_input=SpecialType(2), num_op_output=Operation(3),
# 																	  missing_values=missing_values_list_parameter_num_op_impute_mean,
# 																	  axis_param=0, field = 'distance')
# 		missing_values_list_parameter_num_op_impute_mean=[]
# 		data_model_impute_outlier_closest_in_transformed=transformations.transform_special_value_num_op(data_dictionary=data_model_impute_outlier_closest_in_transformed,
# 																	  special_type_input=SpecialType(2), num_op_output=Operation(3),
# 																	  missing_values=missing_values_list_parameter_num_op_impute_mean,
# 																	  axis_param=0, field = 'premiere')
# 		missing_values_list_parameter_num_op_impute_mean=[]
# 		data_model_impute_outlier_closest_in_transformed=transformations.transform_special_value_num_op(data_dictionary=data_model_impute_outlier_closest_in_transformed,
# 																	  special_type_input=SpecialType(2), num_op_output=Operation(3),
# 																	  missing_values=missing_values_list_parameter_num_op_impute_mean,
# 																	  axis_param=0, field = 'sex')
# 		missing_values_list_parameter_num_op_impute_mean=[]
# 		data_model_impute_outlier_closest_in_transformed=transformations.transform_special_value_num_op(data_dictionary=data_model_impute_outlier_closest_in_transformed,
# 																	  special_type_input=SpecialType(2), num_op_output=Operation(3),
# 																	  missing_values=missing_values_list_parameter_num_op_impute_mean,
# 																	  axis_param=0, field = 'Enroll')
# 		missing_values_list_parameter_num_op_impute_mean=[]
# 		data_model_impute_outlier_closest_in_transformed=transformations.transform_special_value_num_op(data_dictionary=data_model_impute_outlier_closest_in_transformed,
# 																	  special_type_input=SpecialType(2), num_op_output=Operation(3),
# 																	  missing_values=missing_values_list_parameter_num_op_impute_mean,
# 																	  axis_param=0, field = 'Instate')
#
# 		data_model_impute_outlier_closest_out=data_model_impute_outlier_closest_in_transformed
# 		data_model_impute_outlier_closest_out.to_csv('./workflow_datasets/data_model_imputeOutliers_out.csv')
#
# #-----------------New DataProcessing-----------------
# 		data_model_binner_in=pd.read_csv('./workflow_datasets/data_model_imputeOutliers_out.csv')
# 		data_model_binner_in_transformed=data_model_binner_in.copy()
# 		data_model_binner_in_transformed=transformations.transform_derived_field(data_dictionary=data_model_binner_in_transformed,
# 																	  data_type_output = DataType(0),
# 																	  field_in = 'TOTAL_CONTACTS', field_out = 'TOTAL_CONTACTS_binned')
# 		data_model_binner_in_transformed=transformations.transform_derived_field(data_dictionary=data_model_binner_in_transformed,
# 																	  data_type_output = DataType(0),
# 																	  field_in = 'SELF_INIT_CNTCTS', field_out = 'SELF_INIT_CNTCTS_binned')
# 		data_model_binner_in_transformed=transformations.transform_derived_field(data_dictionary=data_model_binner_in_transformed,
# 																	  data_type_output = DataType(0),
# 																	  field_in = 'SOLICITED_CNTCTS', field_out = 'SOLICITED_CNTCTS_binned')
#
# 		data_model_binner_in_transformed=transformations.transform_interval_fix_value(data_dictionary=data_model_binner_in_transformed,
# 																	  left_margin=-1000.0, right_margin=1.0,
# 																	  closure_type=Closure(0),
# 																	  fix_value_output='low',
# 								                                      data_type_output = DataType(0),
# 																	  field_in = 'TOTAL_CONTACTS',
# 																	  field_out = 'TOTAL_CONTACTS_binned')
#
# 		data_model_binner_in_transformed=transformations.transform_interval_fix_value(data_dictionary=data_model_binner_in_transformed,
# 																	  left_margin=-1000.0, right_margin=1.0,
# 																	  closure_type=Closure(0),
# 																	  fix_value_output='low',
# 								                                      data_type_output = DataType(0),
# 																	  field_in = 'SELF_INIT_CNTCTS',
# 																	  field_out = 'SELF_INIT_CNTCTS_binned')
#
# 		data_model_binner_in_transformed=transformations.transform_interval_fix_value(data_dictionary=data_model_binner_in_transformed,
# 																	  left_margin=-1000.0, right_margin=1.0,
# 																	  closure_type=Closure(0),
# 																	  fix_value_output='low',
# 								                                      data_type_output = DataType(0),
# 																	  field_in = 'SOLICITED_CNTCTS',
# 																	  field_out = 'SOLICITED_CNTCTS_binned')
# 		data_model_binner_out=data_model_binner_in_transformed
# 		data_model_binner_out.to_csv('./workflow_datasets/data_model_binner1_out.csv')
#
# 		data_model_binner_in_transformed=transformations.transform_interval_fix_value(data_dictionary=data_model_binner_in_transformed,
# 																	  left_margin=1.0, right_margin=4.0,
# 																	  closure_type=Closure(2),
# 																	  fix_value_output='Moderate',
# 								                                      data_type_output = DataType(0),
# 																	  field_in = 'TOTAL_CONTACTS',
# 																	  field_out = 'TOTAL_CONTACTS_binned')
#
# 		data_model_binner_in_transformed=transformations.transform_interval_fix_value(data_dictionary=data_model_binner_in_transformed,
# 																	  left_margin=1.0, right_margin=4.0,
# 																	  closure_type=Closure(2),
# 																	  fix_value_output='Moderate',
# 								                                      data_type_output = DataType(0),
# 																	  field_in = 'SELF_INIT_CNTCTS',
# 																	  field_out = 'SELF_INIT_CNTCTS_binned')
#
# 		data_model_binner_in_transformed=transformations.transform_interval_fix_value(data_dictionary=data_model_binner_in_transformed,
# 																	  left_margin=1.0, right_margin=4.0,
# 																	  closure_type=Closure(2),
# 																	  fix_value_output='Moderate',
# 								                                      data_type_output = DataType(0),
# 																	  field_in = 'SOLICITED_CNTCTS',
# 																	  field_out = 'SOLICITED_CNTCTS_binned')
# 		data_model_binner_out=data_model_binner_in_transformed
# 		data_model_binner_out.to_csv('./workflow_datasets/data_model_binner1_out.csv')
#
# 		data_model_binner_in_transformed=transformations.transform_interval_fix_value(data_dictionary=data_model_binner_in_transformed,
# 																	  left_margin=4.0, right_margin=1000.0,
# 																	  closure_type=Closure(2),
# 																	  fix_value_output='High',
# 								                                      data_type_output = DataType(0),
# 																	  field_in = 'TOTAL_CONTACTS',
# 																	  field_out = 'TOTAL_CONTACTS_binned')
#
# 		data_model_binner_in_transformed=transformations.transform_interval_fix_value(data_dictionary=data_model_binner_in_transformed,
# 																	  left_margin=4.0, right_margin=1000.0,
# 																	  closure_type=Closure(2),
# 																	  fix_value_output='High',
# 								                                      data_type_output = DataType(0),
# 																	  field_in = 'SELF_INIT_CNTCTS',
# 																	  field_out = 'SELF_INIT_CNTCTS_binned')
#
# 		data_model_binner_in_transformed=transformations.transform_interval_fix_value(data_dictionary=data_model_binner_in_transformed,
# 																	  left_margin=4.0, right_margin=1000.0,
# 																	  closure_type=Closure(2),
# 																	  fix_value_output='High',
# 								                                      data_type_output = DataType(0),
# 																	  field_in = 'SOLICITED_CNTCTS',
# 																	  field_out = 'SOLICITED_CNTCTS_binned')
# 		data_model_binner_out=data_model_binner_in_transformed
# 		data_model_binner_out.to_csv('./workflow_datasets/data_model_binner1_out.csv')
#
# #-----------------New DataProcessing-----------------
# 		data_model_binner_in=pd.read_csv('./workflow_datasets/data_model_binner1_out.csv')
# 		data_model_binner_in_transformed=data_model_binner_in.copy()
# 		data_model_binner_in_transformed=transformations.transform_derived_field(data_dictionary=data_model_binner_in_transformed,
# 																	  data_type_output = DataType(0),
# 																	  field_in = 'TERRITORY', field_out = 'TERRITORY_binned')
#
# 		data_model_binner_in_transformed=transformations.transform_interval_fix_value(data_dictionary=data_model_binner_in_transformed,
# 																	  left_margin=-1000.0, right_margin=1.0,
# 																	  closure_type=Closure(0),
# 																	  fix_value_output='Unknown',
# 								                                      data_type_output = DataType(0),
# 																	  field_in = 'TERRITORY',
# 																	  field_out = 'TERRITORY_binned')
# 		data_model_binner_out=data_model_binner_in_transformed
# 		data_model_binner_out.to_csv('./workflow_datasets/data_model_binner2_out.csv')
#
# 		data_model_binner_in_transformed=transformations.transform_interval_fix_value(data_dictionary=data_model_binner_in_transformed,
# 																	  left_margin=1.0, right_margin=3.0,
# 																	  closure_type=Closure(2),
# 																	  fix_value_output='Zone 1',
# 								                                      data_type_output = DataType(0),
# 																	  field_in = 'TERRITORY',
# 																	  field_out = 'TERRITORY_binned')
# 		data_model_binner_out=data_model_binner_in_transformed
# 		data_model_binner_out.to_csv('./workflow_datasets/data_model_binner2_out.csv')
#
# 		data_model_binner_in_transformed=transformations.transform_interval_fix_value(data_dictionary=data_model_binner_in_transformed,
# 																	  left_margin=3.0, right_margin=5.0,
# 																	  closure_type=Closure(2),
# 																	  fix_value_output='Zone 2',
# 								                                      data_type_output = DataType(0),
# 																	  field_in = 'TERRITORY',
# 																	  field_out = 'TERRITORY_binned')
# 		data_model_binner_out=data_model_binner_in_transformed
# 		data_model_binner_out.to_csv('./workflow_datasets/data_model_binner2_out.csv')
#
# 		data_model_binner_in_transformed=transformations.transform_interval_fix_value(data_dictionary=data_model_binner_in_transformed,
# 																	  left_margin=5.0, right_margin=7.0,
# 																	  closure_type=Closure(2),
# 																	  fix_value_output='Zone 3',
# 								                                      data_type_output = DataType(0),
# 																	  field_in = 'TERRITORY',
# 																	  field_out = 'TERRITORY_binned')
# 		data_model_binner_out=data_model_binner_in_transformed
# 		data_model_binner_out.to_csv('./workflow_datasets/data_model_binner2_out.csv')
#
# 		data_model_binner_in_transformed=transformations.transform_interval_fix_value(data_dictionary=data_model_binner_in_transformed,
# 																	  left_margin=7.0, right_margin=1000.0,
# 																	  closure_type=Closure(3),
# 																	  fix_value_output='Zone 4',
# 								                                      data_type_output = DataType(0),
# 																	  field_in = 'TERRITORY',
# 																	  field_out = 'TERRITORY_binned')
# 		data_model_binner_out=data_model_binner_in_transformed
# 		data_model_binner_out.to_csv('./workflow_datasets/data_model_binner2_out.csv')
		















dp=DataProcessing()
dp.generateDataProcessing()
