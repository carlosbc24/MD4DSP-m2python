import pandas as pd
import functions.contract_invariants as contract_invariants
import functions.contract_pre_post as contract_pre_post
from helpers.enumerations import Belong, Operator, Operation, SpecialType, DataType, DerivedType, Closure


class DataProcessing:
	def generateDataProcessing(self):
		pre_post=contract_pre_post.ContractsPrePost()
		invariants=contract_invariants.Invariants()
#-----------------New DataProcessing-----------------
		data_model_impute_sex_in=pd.read_csv('../data_model.csv')

		missing_values_PRE_value_range_impute_sex=[]
		if pre_post.check_missing_range(belong_op=Belong(0), data_dictionary=data_model_impute_sex_in, field='sex', 
										missing_values=missing_values_PRE_value_range_impute_sex,
										quant_op=Operator(2), quant_rel=70.0/100):
			print('PRECONDITION PRE_value_range_impute_sex VALIDATED')
		else:
			print('PRECONDITION PRE_value_range_impute_sex NOT VALIDATED')
		

#-----------------New DataProcessing-----------------
		data_model_impute_IRSCHOOL_in=pd.read_csv('../data_model.csv')

		missing_values_PRE_value_range_impute_IRSCHOOL=[]
		if pre_post.check_missing_range(belong_op=Belong(0), data_dictionary=data_model_impute_IRSCHOOL_in, field='IRSCHOOL', 
										missing_values=missing_values_PRE_value_range_impute_IRSCHOOL,
										quant_op=Operator(2), quant_rel=30.0/100):
			print('PRECONDITION PRE_value_range_impute_IRSCHOOL VALIDATED')
		else:
			print('PRECONDITION PRE_value_range_impute_IRSCHOOL NOT VALIDATED')
		
		

#-----------------New DataProcessing-----------------
		data_model_impute_ETHNICITY_in=pd.read_csv('../data_model.csv')

		invalid_values_PRE_value_range_impute_ETHNICITY=[14]
		if pre_post.check_invalid_values(belong_op=Belong(0), data_dictionary=data_model_impute_ETHNICITY_in, field='ETHNICITY', 
										invalid_values=invalid_values_PRE_value_range_impute_ETHNICITY,
										quant_op=Operator(3), quant_rel=30.0/100):
			print('PRECONDITION PRE_value_range_impute_ETHNICITY VALIDATED')
		else:
			print('PRECONDITION PRE_value_range_impute_ETHNICITY NOT VALIDATED')
		

		
#-----------------New DataProcessing-----------------
		data_model_impute_in=pd.read_csv('../data_model.csv')

		missing_values_PRE_value_range_impute_sex_columns=[4]
		if pre_post.check_missing_range(belong_op=Belong(0), data_dictionary=data_model_impute_in, field='sex', 
										missing_values=missing_values_PRE_value_range_impute_sex_columns,
										quant_op=Operator(2), quant_rel=30.0/100):
			print('PRECONDITION PRE_value_range_impute_sex_columns VALIDATED')
		else:
			print('PRECONDITION PRE_value_range_impute_sex_columns NOT VALIDATED')
		
		missing_values_PRE_value_range_impute_IRSCHOOL_columns=[]
		if pre_post.check_missing_range(belong_op=Belong(0), data_dictionary=data_model_impute_in, field='IRSCHOOL', 
										missing_values=missing_values_PRE_value_range_impute_IRSCHOOL_columns,
										quant_op=Operator(2), quant_rel=30.0/100):
			print('PRECONDITION PRE_value_range_impute_IRSCHOOL_columns VALIDATED')
		else:
			print('PRECONDITION PRE_value_range_impute_IRSCHOOL_columns NOT VALIDATED')
		
		missing_values_PRE_value_range_impute_ETHNICITY_columns=[]
		if pre_post.check_missing_range(belong_op=Belong(0), data_dictionary=data_model_impute_in, field='ETHNICITY', 
										missing_values=missing_values_PRE_value_range_impute_ETHNICITY_columns,
										quant_op=Operator(2), quant_rel=30.0/100):
			print('PRECONDITION PRE_value_range_impute_ETHNICITY_columns VALIDATED')
		else:
			print('PRECONDITION PRE_value_range_impute_ETHNICITY_columns NOT VALIDATED')
		

#-----------------New DataProcessing-----------------
		data_model_impute_ACADEMIC_INTEREST_2_in=pd.read_csv('../data_model.csv')

		missing_values_PRE_value_range_impute_ACADEMIC_INTEREST_2=[]
		if pre_post.check_missing_range(belong_op=Belong(0), data_dictionary=data_model_impute_ACADEMIC_INTEREST_2_in, field='ACADEMIC_INTEREST_2', 
										missing_values=missing_values_PRE_value_range_impute_ACADEMIC_INTEREST_2,
										quant_op=Operator(2), quant_rel=30.0/100):
			print('PRECONDITION PRE_value_range_impute_ACADEMIC_INTEREST_2 VALIDATED')
		else:
			print('PRECONDITION PRE_value_range_impute_ACADEMIC_INTEREST_2 NOT VALIDATED')
		
		missing_values_PRE_value_range_impute_ACADEMIC_INTEREST_1=[]
		if pre_post.check_missing_range(belong_op=Belong(0), data_dictionary=data_model_impute_ACADEMIC_INTEREST_2_in, field='ACADEMIC_INTEREST_1', 
										missing_values=missing_values_PRE_value_range_impute_ACADEMIC_INTEREST_1,
										quant_op=Operator(2), quant_rel=30.0/100):
			print('PRECONDITION PRE_value_range_impute_ACADEMIC_INTEREST_1 VALIDATED')
		else:
			print('PRECONDITION PRE_value_range_impute_ACADEMIC_INTEREST_1 NOT VALIDATED')

#-----------------New DataProcessing-----------------
		data_model_impute_mean_in=pd.read_csv('../data_model.csv')

		missing_values_PRE_value_range_impute_mean_avg_income=[]
		if pre_post.check_missing_range(belong_op=Belong(0), data_dictionary=data_model_impute_mean_in, field='avg_income', 
										missing_values=missing_values_PRE_value_range_impute_mean_avg_income,
										quant_abs=None, quant_rel=None, quant_op=None):
			print('PRECONDITION PRE_value_range_impute_mean_avg_income VALIDATED')
		else:
			print('PRECONDITION PRE_value_range_impute_mean_avg_income NOT VALIDATED')
		
		missing_values_PRE_value_range_impute_mean_distance=[]
		if pre_post.check_missing_range(belong_op=Belong(0), data_dictionary=data_model_impute_mean_in, field='distance', 
										missing_values=missing_values_PRE_value_range_impute_mean_distance,
										quant_abs=None, quant_rel=None, quant_op=None):
			print('PRECONDITION PRE_value_range_impute_mean_distance VALIDATED')
		else:
			print('PRECONDITION PRE_value_range_impute_mean_distance NOT VALIDATED')
		
		

#-----------------New DataProcessing-----------------
		data_model_impute_linear_interpolation_in=pd.read_csv('../data_model.csv')

		missing_values_PRE_value_range_impute_linear_interpolation_satscore=[]
		if pre_post.check_missing_range(belong_op=Belong(0), data_dictionary=data_model_impute_linear_interpolation_in, field='satscore', 
										missing_values=missing_values_PRE_value_range_impute_linear_interpolation_satscore,
										quant_abs=None, quant_rel=None, quant_op=None):
			print('PRECONDITION PRE_value_range_impute_linear_interpolation_satscore VALIDATED')
		else:
			print('PRECONDITION PRE_value_range_impute_linear_interpolation_satscore NOT VALIDATED')
		
		

		
#-----------------New DataProcessing-----------------
		#
		# if pre_post.check_fix_value_range(value=0, data_dictionary=data_model_row_filter_in, belong_op=Belong(0), field='init_span',
		# 								quant_abs=None, quant_rel=None, quant_op=None):
		# 	print('PRECONDITION PRE_value_range_row_filter VALIDATED')
		# else:
		# 	print('PRECONDITION PRE_value_range_row_filter NOT VALIDATED')
		#
		
		
		

#-----------------New DataProcessing-----------------
		# data_model_column_cont_filter_in=pd.read_csv('../workflow_datasets/data_model_row_filter_out.csv')
		#
		# field_list_PRE_field_range_column_cont_filter=['TRAVEL_INIT_CNTCTS', 'REFERRAL_CNCTS', 'telecq', 'stuemail', 'interest']
		# if pre_post.check_field_range(fields=field_list_PRE_field_range_column_cont_filter,
		# 							data_dictionary=data_model_column_cont_filter_in,
		# 							belong_op=Belong(0)):
		# 	print('PRECONDITION PRE_field_range_column_cont_filter VALIDATED')
		# else:
		# 	print('PRECONDITION PRE_field_range_column_cont_filter NOT VALIDATED')
		#
		
		

#-----------------New DataProcessing-----------------
		# data_model_column_cat_filter_in=pd.read_csv('../workflow_datasets/data_model_row_filter_out.csv')
		#
		# field_list_PRE_field_range_column_cat_filter=['CONTACT_CODE1']
		# if pre_post.check_field_range(fields=field_list_PRE_field_range_column_cat_filter,
		# 							data_dictionary=data_model_column_cat_filter_in,
		# 							belong_op=Belong(0)):
		# 	print('PRECONDITION PRE_field_range_column_cat_filter VALIDATED')
		# else:
		# 	print('PRECONDITION PRE_field_range_column_cat_filter NOT VALIDATED')
		#

#-----------------New DataProcessing-----------------
		# data_model_map_territory_in=pd.read_csv('../workflow_datasets/data_model_col_filter_out.csv')
		#
		# if pre_post.check_fix_value_range(value='A', data_dictionary=data_model_map_territory_in, belong_op=Belong(0), field='TERRITORY',
		# 								quant_abs=None, quant_rel=None, quant_op=None):
		# 	print('PRECONDITION PRE_value_range_territory VALIDATED')
		# else:
		# 	print('PRECONDITION PRE_value_range_territory NOT VALIDATED')
		# if pre_post.check_fix_value_range(value='N', data_dictionary=data_model_map_territory_in, belong_op=Belong(0), field='TERRITORY',
		# 								quant_abs=None, quant_rel=None, quant_op=None):
		# 	print('PRECONDITION PRE_value_range_territory VALIDATED')
		# else:
		# 	print('PRECONDITION PRE_value_range_territory NOT VALIDATED')
		#
		# print('Transformation of type FixValue-FixValue')
		#
		#
		# print('Transformation of type FixValue-FixValue')

#-----------------New DataProcessing-----------------
# 		data_model_map_Instate_in=pd.read_csv('../workflow_datasets/data_model_map_territory_out.csv')
#
# 		if pre_post.check_fix_value_range(value='Y', data_dictionary=data_model_map_Instate_in, belong_op=Belong(0), field='Instate',
# 										quant_abs=None, quant_rel=None, quant_op=None):
# 			print('PRECONDITION PRE_value_range_Instate VALIDATED')
# 		else:
# 			print('PRECONDITION PRE_value_range_Instate NOT VALIDATED')
# 		if pre_post.check_fix_value_range(value='N', data_dictionary=data_model_map_Instate_in, belong_op=Belong(0), field='Instate',
# 										quant_abs=None, quant_rel=None, quant_op=None):
# 			print('PRECONDITION PRE_value_range_Instate VALIDATED')
# 		else:
# 			print('PRECONDITION PRE_value_range_Instate NOT VALIDATED')
#
#
# 		print('Transformation of type FixValue-FixValue')
#
#
# 		print('Transformation of type FixValue-FixValue')
#
# #-----------------New DataProcessing-----------------
# 		data_model_stringToNumber_in=pd.read_csv('../workflow_datasets/data_model_map_instate_out.csv')
#
#
#
#
#
#
#
#
		
		
		
#-----------------New DataProcessing-----------------
# 		data_model_impute_outlier_closest_in=pd.read_csv('../workflow_datasets/data_model_stringToNumber_in.csv')
#
# 		if pre_post.check_outliers(belong_op=Belong(0), data_dictionary=data_model_impute_outlier_closest_in, field='avg_income',
# 										quant_abs=None, quant_rel=None, quant_op=None):
# 			print('PRECONDITION PRE_value_range_impute_outliers_closest_avg_income VALIDATED')
# 		else:
# 			print('PRECONDITION PRE_value_range_impute_outliers_closest_avg_income NOT VALIDATED')
#
# 		if pre_post.check_outliers(belong_op=Belong(0), data_dictionary=data_model_impute_outlier_closest_in, field='distance',
# 										quant_abs=None, quant_rel=None, quant_op=None):
# 			print('PRECONDITION PRE_value_range_impute_outliers_closest_distance VALIDATED')
# 		else:
# 			print('PRECONDITION PRE_value_range_impute_outliers_closest_distance NOT VALIDATED')
#
# 		if pre_post.check_outliers(belong_op=Belong(0), data_dictionary=data_model_impute_outlier_closest_in, field='premiere',
# 										quant_abs=None, quant_rel=None, quant_op=None):
# 			print('PRECONDITION PRE_value_range_impute_outliers_closest_premiere VALIDATED')
# 		else:
# 			print('PRECONDITION PRE_value_range_impute_outliers_closest_premiere NOT VALIDATED')
#
# 		if pre_post.check_outliers(belong_op=Belong(0), data_dictionary=data_model_impute_outlier_closest_in, field='sex',
# 										quant_abs=None, quant_rel=None, quant_op=None):
# 			print('PRECONDITION PRE_value_range_impute_outliers_closest_sex VALIDATED')
# 		else:
# 			print('PRECONDITION PRE_value_range_impute_outliers_closest_sex NOT VALIDATED')
#
# 		if pre_post.check_outliers(belong_op=Belong(0), data_dictionary=data_model_impute_outlier_closest_in, field='Enroll',
# 										quant_abs=None, quant_rel=None, quant_op=None):
# 			print('PRECONDITION PRE_value_range_impute_outliers_closest_Enroll VALIDATED')
# 		else:
# 			print('PRECONDITION PRE_value_range_impute_outliers_closest_Enroll NOT VALIDATED')
#
# 		if pre_post.check_outliers(belong_op=Belong(0), data_dictionary=data_model_impute_outlier_closest_in, field='Instate',
# 										quant_abs=None, quant_rel=None, quant_op=None):
# 			print('PRECONDITION PRE_value_range_impute_outliers_closest_Instate VALIDATED')
# 		else:
# 			print('PRECONDITION PRE_value_range_impute_outliers_closest_Instate NOT VALIDATED')
#
#
#
#
# #-----------------New DataProcessing-----------------
# 		data_model_binner_in=pd.read_csv('../workflow_datasets/data_model_stringToNumber_in.csv')
#
# 		if pre_post.check_interval_range_float(left_margin=-1000.0, right_margin=1.0, data_dictionary=data_model_binner_in,
# 		                                	closure_type=Closure(0), belong_op=Belong(0), field='TOTAL_CONTACTS'):
# 			print('PRECONDITION PRE_binner_valueRange_TOTAL_CONTACTS VALIDATED')
# 		else:
# 			print('PRECONDITION PRE_binner_valueRange_TOTAL_CONTACTS NOT VALIDATED')
#
# 		if pre_post.check_interval_range_float(left_margin=-1000.0, right_margin=1.0, data_dictionary=data_model_binner_in,
# 		                                	closure_type=Closure(0), belong_op=Belong(0), field='SELF_INIT_CNTCTS'):
# 			print('PRECONDITION PRE_binner_valueRange_SELF_INIT_CNTCTS VALIDATED')
# 		else:
# 			print('PRECONDITION PRE_binner_valueRange_SELF_INIT_CNTCTS NOT VALIDATED')
#
# 		if pre_post.check_interval_range_float(left_margin=-1000.0, right_margin=1.0, data_dictionary=data_model_binner_in,
# 		                                	closure_type=Closure(0), belong_op=Belong(0), field='SOLICITED_CNTCTS'):
# 			print('PRECONDITION PRE_binner_valueRange_SOLICITED_CNTCTS VALIDATED')
# 		else:
# 			print('PRECONDITION PRE_binner_valueRange_SOLICITED_CNTCTS NOT VALIDATED')
#
#
#
#
#
# #-----------------New DataProcessing-----------------
# 		data_model_binner_in=pd.read_csv('../workflow_datasets/data_model_stringToNumber_in.csv')
#
# 		if pre_post.check_interval_range_float(left_margin=0.0, right_margin=1000.0, data_dictionary=data_model_binner_in,
# 		                                	closure_type=Closure(3), belong_op=Belong(0), field='TERRITORY'):
# 			print('PRECONDITION PRE_binner_valueRange_TERRITORY VALIDATED')
# 		else:
# 			print('PRECONDITION PRE_binner_valueRange_TERRITORY NOT VALIDATED')






dp=DataProcessing()
dp.generateDataProcessing()
