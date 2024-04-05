import pandas as pd
import functions.contract_invariants as contract_invariants
import functions.contract_pre_post as contract_pre_post
from helpers.enumerations import Belong, Operator, Operation, SpecialType, DataType
class DataProcessing:
	def generateDataProcessing(self):
		pre_post=contract_pre_post.ContractsPrePost()
		invariants=contract_invariants.ContractsInvariants()
#-----------------New DataProcessing-----------------
		data_model_impute_in=pd.read_csv('../data_model.csv')
		data_model_impute_in_copy=data_model_impute_in.copy()
		
		missing_values_PRE_value_range_impute_sex_columns=[4]
		if pre_post.checkMissingRange(belongOp=Belong(0), dataDictionary=data_model_impute_in, field='sex', 
										missing_values=missing_values_PRE_value_range_impute_sex_columns,
										quant_op=Operator(2), quant_rel=30.0/100):
			print('PRECONDITION call returned TRUE')
		else:
			print('PRECONDITION call returned FALSE')
		
		
		
		missing_values_PRE_value_range_impute_IRSCHOOL_columns=[]
		if pre_post.checkMissingRange(belongOp=Belong(0), dataDictionary=data_model_impute_in, field='IRSCHOOL', 
										missing_values=missing_values_PRE_value_range_impute_IRSCHOOL_columns,
										quant_op=Operator(2), quant_rel=30.0/100):
			print('PRECONDITION call returned TRUE')
		else:
			print('PRECONDITION call returned FALSE')
		
		
		
		missing_values_PRE_value_range_impute_ETHNICITY_columns=[]
		if pre_post.checkMissingRange(belongOp=Belong(0), dataDictionary=data_model_impute_in, field='ETHNICITY', 
										missing_values=missing_values_PRE_value_range_impute_ETHNICITY_columns,
										quant_op=Operator(2), quant_rel=30.0/100):
			print('PRECONDITION call returned TRUE')
		else:
			print('PRECONDITION call returned FALSE')
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		# missing_values_POST_value_range_impute_sex_columns=[]
		# if pre_post.checkMissingRange(belongOp=Belong(1), dataDictionary=data_model_impute_out, field='sex',
		# 								missing_values=missing_values_POST_value_range_impute_sex_columns,
		# 								quant_abs=None, quant_rel=None, quant_op=None):
		# 	print('POSTCONDITION call returned TRUE')
		# else:
		# 	print('POSTCONDITION call returned FALSE')
		#
		#
		#
		# missing_values_POST_value_range_impute_IRSCHOOL_columns=[]
		# if pre_post.checkMissingRange(belongOp=Belong(1), dataDictionary=data_model_impute_out, field='IRSCHOOL',
		# 								missing_values=missing_values_POST_value_range_impute_IRSCHOOL_columns,
		# 								quant_abs=None, quant_rel=None, quant_op=None):
		# 	print('POSTCONDITION call returned TRUE')
		# else:
		# 	print('POSTCONDITION call returned FALSE')
		#
		#
		#
		# missing_values_POST_value_range_impute_ETHNICITY_columns=[]
		# if pre_post.checkMissingRange(belongOp=Belong(1), dataDictionary=data_model_impute_ETHNICITY_out, field='ETHNICITY',
		# 								missing_values=missing_values_POST_value_range_impute_ETHNICITY_columns,
		# 								quant_abs=None, quant_rel=None, quant_op=None):
		# 	print('POSTCONDITION call returned TRUE')
		# else:
		# 	print('POSTCONDITION call returned FALSE')
		
		
#-----------------New DataProcessing-----------------
		data_model_impute_sex_in=pd.read_csv('../data_model.csv')
		data_model_impute_sex_in_copy=data_model_impute_sex_in.copy()
		
		missing_values_PRE_value_range_impute_sex=[]
		if pre_post.checkMissingRange(belongOp=Belong(0), dataDictionary=data_model_impute_sex_in, field='sex', 
										missing_values=missing_values_PRE_value_range_impute_sex,
										quant_op=Operator(2), quant_rel=70.0/100):
			print('PRECONDITION call returned TRUE')
		else:
			print('PRECONDITION call returned FALSE')
		
		
		
		
		
		
		
		
		
		# missing_values_POST_value_range_impute_sex=[]
		# if pre_post.checkMissingRange(belongOp=Belong(1), dataDictionary=data_model_impute_sex_out, field='sex',
		# 								missing_values=missing_values_POST_value_range_impute_sex,
		# 								quant_abs=None, quant_rel=None, quant_op=None):
		# 	print('POSTCONDITION call returned TRUE')
		# else:
		# 	print('POSTCONDITION call returned FALSE')
		
		
#-----------------New DataProcessing-----------------
		data_model_impute_IRSCHOOL_in=pd.read_csv('../data_model.csv')
		data_model_impute_IRSCHOOL_in_copy=data_model_impute_IRSCHOOL_in.copy()
		
		missing_values_PRE_value_range_impute_IRSCHOOL=[]
		if pre_post.checkMissingRange(belongOp=Belong(0), dataDictionary=data_model_impute_IRSCHOOL_in, field='IRSCHOOL', 
										missing_values=missing_values_PRE_value_range_impute_IRSCHOOL,
										quant_op=Operator(2), quant_rel=30.0/100):
			print('PRECONDITION call returned TRUE')
		else:
			print('PRECONDITION call returned FALSE')
		
		
		
		
		
		
		
		
		
		# missing_values_POST_value_range_impute_IRSCHOOL=[]
		# if pre_post.checkMissingRange(belongOp=Belong(1), dataDictionary=data_model_impute_IRSCHOOL_out, field='IRSCHOOL',
		# 								missing_values=missing_values_POST_value_range_impute_IRSCHOOL,
		# 								quant_abs=None, quant_rel=None, quant_op=None):
		# 	print('POSTCONDITION call returned TRUE')
		# else:
		# 	print('POSTCONDITION call returned FALSE')
		
		
#-----------------New DataProcessing-----------------
		data_model_impute_ETHNICITY_in=pd.read_csv('../data_model.csv')
		data_model_impute_ETHNICITY_in_copy=data_model_impute_ETHNICITY_in.copy()
		
		invalid_values_PRE_value_range_impute_ETHNICITY=[14]
		if pre_post.checkInvalidValues(belongOp=Belong(0), dataDictionary=data_model_impute_ETHNICITY_in, field='ETHNICITY', 
										invalid_values=invalid_values_PRE_value_range_impute_ETHNICITY,
										quant_op=Operator(3), quant_rel=30.0/100):
			print('PRECONDITION call returned TRUE')
		else:
			print('PRECONDITION call returned FALSE')
		
		
		
		
		
		
		
		
		
		# missing_values_POST_value_range_impute_ETHNICITY=[]
		# if pre_post.checkMissingRange(belongOp=Belong(1), dataDictionary=data_model_impute_ETHNICITY_out, field='ETHNICITY',
		# 								missing_values=missing_values_POST_value_range_impute_ETHNICITY,
		# 								quant_abs=None, quant_rel=None, quant_op=None):
		# 	print('POSTCONDITION call returned TRUE')
		# else:
		# 	print('POSTCONDITION call returned FALSE')
		
		
#-----------------New DataProcessing-----------------
		data_model_impute_ACADEMIC_INTEREST_2_in=pd.read_csv('../data_model.csv')
		data_model_impute_ACADEMIC_INTEREST_2_in_copy=data_model_impute_ACADEMIC_INTEREST_2_in.copy()
		
		missing_values_PRE_value_range_impute_ACADEMIC_INTEREST_2=[]
		if pre_post.checkMissingRange(belongOp=Belong(0), dataDictionary=data_model_impute_ACADEMIC_INTEREST_2_in, field='ACADEMIC_INTEREST_2', 
										missing_values=missing_values_PRE_value_range_impute_ACADEMIC_INTEREST_2,
										quant_op=Operator(2), quant_rel=30.0/100):
			print('PRECONDITION call returned TRUE')
		else:
			print('PRECONDITION call returned FALSE')
		
		
		
		missing_values_PRE_value_range_impute_ACADEMIC_INTEREST_1=[]
		if pre_post.checkMissingRange(belongOp=Belong(0), dataDictionary=data_model_impute_ACADEMIC_INTEREST_2_in, field='ACADEMIC_INTEREST_1', 
										missing_values=missing_values_PRE_value_range_impute_ACADEMIC_INTEREST_1,
										quant_op=Operator(2), quant_rel=30.0/100):
			print('PRECONDITION call returned TRUE')
		else:
			print('PRECONDITION call returned FALSE')
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		# missing_values_POST_value_range_impute_ACADEMIC_INTEREST_2=[]
		# if pre_post.checkMissingRange(belongOp=Belong(1), dataDictionary=data_model_impute_ACADEMIC_INTEREST_2_out, field='ACADEMIC_INTEREST_2',
		# 								missing_values=missing_values_POST_value_range_impute_ACADEMIC_INTEREST_2,
		# 								quant_abs=None, quant_rel=None, quant_op=None):
		# 	print('POSTCONDITION call returned TRUE')
		# else:
		# 	print('POSTCONDITION call returned FALSE')
		#
		#
		#
		# missing_values_POST_value_range_impute_ACADEMIC_INTEREST_1=[]
		# if pre_post.checkMissingRange(belongOp=Belong(1), dataDictionary=data_model_impute_ACADEMIC_INTEREST_2_out, field='ACADEMIC_INTEREST_1',
		# 								missing_values=missing_values_POST_value_range_impute_ACADEMIC_INTEREST_1,
		# 								quant_abs=None, quant_rel=None, quant_op=None):
		# 	print('POSTCONDITION call returned TRUE')
		# else:
		# 	print('POSTCONDITION call returned FALSE')
		#
		
#-----------------New DataProcessing-----------------
		data_model_impute_mean_in=pd.read_csv('../data_model.csv')
		data_model_impute_mean_in_copy=data_model_impute_mean_in.copy()
		
		missing_values_PRE_value_range_impute_mean_avg_income=[]
		if pre_post.checkMissingRange(belongOp=Belong(0), dataDictionary=data_model_impute_mean_in, field='avg_income', 
										missing_values=missing_values_PRE_value_range_impute_mean_avg_income,
										quant_abs=None, quant_rel=None, quant_op=None):
			print('PRECONDITION call returned TRUE')
		else:
			print('PRECONDITION call returned FALSE')
		
		
		missing_values_dfbdf=None
		if pre_post.checkMissingRange(belongOp=Belong(0), dataDictionary=data_model_impute_mean_in, field=None, 
										missing_values=missing_values_dfbdf,
										quant_abs=None, quant_rel=None, quant_op=None):
			print('PRECONDITION call returned TRUE')
		else:
			print('PRECONDITION call returned FALSE')
		
		
		
		missing_values_PRE_value_range_impute_mean_distance=[]
		if pre_post.checkMissingRange(belongOp=Belong(0), dataDictionary=data_model_impute_mean_in, field='distance', 
										missing_values=missing_values_PRE_value_range_impute_mean_distance,
										quant_abs=None, quant_rel=None, quant_op=None):
			print('PRECONDITION call returned TRUE')
		else:
			print('PRECONDITION call returned FALSE')
		
		
		
		missing_values_INV_condition_avg_income=[]
		#FALTA MIRAR LO DEL BELONG OP DE LAS INVARIANTES
		data_model_impute_mean_in_copy=invariants.checkInv_SpecialValue_NumOp(dataDictionary=data_model_impute_mean_in_copy,
														specialTypeInput=SpecialType(0),
														numOpOutput=Operation(1),
														missing_values=missing_values_INV_condition_avg_income, axis_param=0, field='avg_income')
		print('INVARIANT call')

		data_model_impute_mean_out=data_model_impute_mean_in_copy.copy()
		data_model_impute_mean_out.to_csv('../workflow_datasets/data_model_impute_out.csv')
		
		
		
		
		
		
		missing_values_INV_condition_distance=[]
		#FALTA MIRAR LO DEL BELONG OP DE LAS INVARIANTES
		data_model_impute_mean_in_copy=invariants.checkInv_SpecialValue_NumOp(dataDictionary=data_model_impute_mean_in_copy,
														specialTypeInput=SpecialType(0),
														numOpOutput=Operation(1),
														missing_values=missing_values_INV_condition_distance, axis_param=0, field='distance')
		print('INVARIANT call')

		data_model_impute_mean_out=data_model_impute_mean_in_copy.copy()
		data_model_impute_mean_out.to_csv('../workflow_datasets/data_model_impute_out.csv')
		
		
		
		
		
		
		missing_values_POST_value_range_impute_mean_distance=[]
		if pre_post.checkMissingRange(belongOp=Belong(1), dataDictionary=data_model_impute_mean_out, field='distance', 
										missing_values=missing_values_POST_value_range_impute_mean_distance,
										quant_abs=None, quant_rel=None, quant_op=None):
			print('POSTCONDITION call returned TRUE')
		else:
			print('POSTCONDITION call returned FALSE')
		
		
		
		missing_values_POST_value_range_impute_mean_avg_income=[]
		if pre_post.checkMissingRange(belongOp=Belong(1), dataDictionary=data_model_impute_mean_out, field='avg_income', 
										missing_values=missing_values_POST_value_range_impute_mean_avg_income,
										quant_abs=None, quant_rel=None, quant_op=None):
			print('POSTCONDITION call returned TRUE')
		else:
			print('POSTCONDITION call returned FALSE')
		
		
#-----------------New DataProcessing-----------------
		data_model_impute_linear_interpolation_in=pd.read_csv('../data_model.csv')
		data_model_impute_linear_interpolation_in_copy=data_model_impute_linear_interpolation_in.copy()
		
		missing_values_PRE_value_range_impute_linear_interpolation_satscore=[]
		if pre_post.checkMissingRange(belongOp=Belong(0), dataDictionary=data_model_impute_linear_interpolation_in, field='satscore', 
										missing_values=missing_values_PRE_value_range_impute_linear_interpolation_satscore,
										quant_abs=None, quant_rel=None, quant_op=None):
			print('PRECONDITION call returned TRUE')
		else:
			print('PRECONDITION call returned FALSE')
		
		
		
		missing_values_INV_condition_distance=[]
		#FALTA MIRAR LO DEL BELONG OP DE LAS INVARIANTES
		data_model_impute_linear_interpolation_in_copy=invariants.checkInv_SpecialValue_NumOp(dataDictionary=data_model_impute_linear_interpolation_in_copy,
														specialTypeInput=SpecialType(0),
														numOpOutput=Operation(0),
														missing_values=missing_values_INV_condition_distance, axis_param=0, field='satscore')
		print('INVARIANT call')
		
		data_model_impute_linear_interpolation_out=data_model_impute_linear_interpolation_in_copy.copy()
		data_model_impute_linear_interpolation_out.to_csv('../workflow_datasets/data_model_impute_out.csv')
		
		
		
		
		
		
		missing_values_POST_value_range_impute_linear_interpolation_satscore=[]
		if pre_post.checkMissingRange(belongOp=Belong(1), dataDictionary=data_model_impute_linear_interpolation_out, field='satscore', 
										missing_values=missing_values_POST_value_range_impute_linear_interpolation_satscore,
										quant_abs=None, quant_rel=None, quant_op=None):
			print('POSTCONDITION call returned TRUE')
		else:
			print('POSTCONDITION call returned FALSE')
		
		
#-----------------New DataProcessing-----------------
		data_model_row_filter_in=pd.read_csv('../workflow_datasets/data_model_impute_out.csv')
		data_model_row_filter_in_copy=data_model_row_filter_in.copy()
		
		if pre_post.checkFixValueRange(value=0, dataDictionary=data_model_row_filter_in, belongOp=Belong(0), field='init_span',
										quant_abs=None, quant_rel=None, quant_op=None):
			print('PRECONDITION call returned TRUE')
		else:
			print('PRECONDITION call returned FALSE')
		
		
		
		# if pre_post.checkFixValueRange(value='0', dataDictionary=data_model_row_filter_out, belongOp=Belong(1), field='init_span',
		# 								quant_abs=None, quant_rel=None, quant_op=None):
		# 	print('POSTCONDITION call returned TRUE')
		# else:
		# 	print('POSTCONDITION call returned FALSE')
		
		
#-----------------New DataProcessing-----------------
		data_model_column_cont_filter_in=pd.read_csv('../workflow_datasets/data_model_row_filter_out.csv')
		data_model_column_cont_filter_in_copy=data_model_column_cont_filter_in.copy()
		
		
		
		
#-----------------New DataProcessing-----------------
		data_model_column_cat_filter_in=pd.read_csv('../workflow_datasets/data_model_row_filter_out.csv')
		data_model_column_cat_filter_in_copy=data_model_column_cat_filter_in.copy()
		
		
		
		
#-----------------New DataProcessing-----------------
		data_model_map_territory_in=pd.read_csv('../workflow_datasets/data_model_col_filter_out.csv')
		data_model_map_territory_in_copy=data_model_map_territory_in.copy()
		
		if pre_post.checkFixValueRange(value='A', dataDictionary=data_model_map_territory_in, belongOp=Belong(0), field='TERRITORY',
										quant_abs=None, quant_rel=None, quant_op=None):
			print('PRECONDITION call returned TRUE')
		else:
			print('PRECONDITION call returned FALSE')
		if pre_post.checkFixValueRange(value='N', dataDictionary=data_model_map_territory_in, belongOp=Belong(0), field='TERRITORY',
										quant_abs=None, quant_rel=None, quant_op=None):
			print('PRECONDITION call returned TRUE')
		else:
			print('PRECONDITION call returned FALSE')
		
		
		
		# data_model_map_territory_out=invariants.checkInv_FixValue_FixValue(dataDictionary=data_model_map_territory_in,
		# 												fixValueInput=A, fixValueOutput=0,
		# 												dataTypeInput=DataType(0),
		# 												dataTypeOutput=DataType(0),
		# 												field='TERRITORY')
		# print('PRECONDITION call')
		#
		# data_model_map_territory_out=invariants.checkInv_FixValue_FixValue(dataDictionary=data_model_map_territory_in,
		# 												fixValueInput=N, fixValueOutput=0,
		# 												dataTypeInput=DataType(0),
		# 												dataTypeOutput=DataType(0),
		# 												field='TERRITORY')
		# print('PRECONDITION call')
		
		
		
		
		
		
		# if pre_post.checkFixValueRange(value='A', dataDictionary=data_model_map_territory_out, belongOp=Belong(0), field='TERRITORY',
		# 								quant_abs=None, quant_rel=None, quant_op=None):
		# 	print('POSTCONDITION call returned TRUE')
		# else:
		# 	print('POSTCONDITION call returned FALSE')
		# if pre_post.checkFixValueRange(value='N', dataDictionary=data_model_map_territory_out, belongOp=Belong(0), field='TERRITORY',
		# 								quant_abs=None, quant_rel=None, quant_op=None):
		# 	print('POSTCONDITION call returned TRUE')
		# else:
		# 	print('POSTCONDITION call returned FALSE')
		
		
#-----------------New DataProcessing-----------------
		data_model_map_Instate_in=pd.read_csv('../workflow_datasets/data_model_map_territory_out.csv')
		data_model_map_Instate_in_copy=data_model_map_Instate_in.copy()
		
		if pre_post.checkFixValueRange(value='Y', dataDictionary=data_model_map_Instate_in, belongOp=Belong(0), field='Instate',
										quant_abs=None, quant_rel=None, quant_op=None):
			print('PRECONDITION call returned TRUE')
		else:
			print('PRECONDITION call returned FALSE')
		if pre_post.checkFixValueRange(value='N', dataDictionary=data_model_map_Instate_in, belongOp=Belong(0), field='Instate',
										quant_abs=None, quant_rel=None, quant_op=None):
			print('PRECONDITION call returned TRUE')
		else:
			print('PRECONDITION call returned FALSE')
		
		
		
		# data_model_map_Instate_out=invariants.checkInv_FixValue_FixValue(dataDictionary=data_model_map_Instate_in,
		# 												fixValueInput=Y, fixValueOutput=1,
		# 												dataTypeInput=DataType(0),
		# 												dataTypeOutput=DataType(0),
		# 												field='Instate')
		# print('PRECONDITION call')
		#
		# data_model_map_Instate_out=invariants.checkInv_FixValue_FixValue(dataDictionary=data_model_map_Instate_in,
		# 												fixValueInput=N, fixValueOutput=0,
		# 												dataTypeInput=DataType(0),
		# 												dataTypeOutput=DataType(0),
		# 												field='Instate')
		# print('PRECONDITION call')
		
		
		
		
		
		
		# if pre_post.checkFixValueRange(value='Y', dataDictionary=data_model_map_Instate_out, belongOp=Belong(0), field='Instate',
		# 								quant_abs=None, quant_rel=None, quant_op=None):
		# 	print('POSTCONDITION call returned TRUE')
		# else:
		# 	print('POSTCONDITION call returned FALSE')
		# if pre_post.checkFixValueRange(value='N', dataDictionary=data_model_map_Instate_out, belongOp=Belong(0), field='Instate',
		# 								quant_abs=None, quant_rel=None, quant_op=None):
		# 	print('POSTCONDITION call returned TRUE')
		# else:
		# 	print('POSTCONDITION call returned FALSE')
		
		
#-----------------New DataProcessing-----------------
		data_model_stringToNumber_in=pd.read_csv('../workflow_datasets/data_model_map_instate_out.csv')
		data_model_stringToNumber_in_copy=data_model_stringToNumber_in.copy()
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
#-----------------New DataProcessing-----------------
		data_model_impute_outlier_closest_in=pd.read_csv('../workflow_datasets/data_model_stringToNumber_in.csv')
		data_model_impute_outlier_closest_in_copy=data_model_impute_outlier_closest_in.copy()
		
		if pre_post.checkOutliers(belongOp=Belong(0), dataDictionary=data_model_impute_outlier_closest_in, field='avg_income', 
										quant_abs=None, quant_rel=None, quant_op=None):
			print('PRECONDITION call returned TRUE')
		else:
			print('PRECONDITION call returned FALSE')
		
		
		
		if pre_post.checkOutliers(belongOp=Belong(0), dataDictionary=data_model_impute_outlier_closest_in, field='distance', 
										quant_abs=None, quant_rel=None, quant_op=None):
			print('PRECONDITION call returned TRUE')
		else:
			print('PRECONDITION call returned FALSE')
		
		
		
		if pre_post.checkOutliers(belongOp=Belong(0), dataDictionary=data_model_impute_outlier_closest_in, field='premiere', 
										quant_abs=None, quant_rel=None, quant_op=None):
			print('PRECONDITION call returned TRUE')
		else:
			print('PRECONDITION call returned FALSE')
		
		
		
		if pre_post.checkOutliers(belongOp=Belong(0), dataDictionary=data_model_impute_outlier_closest_in, field='sex', 
										quant_abs=None, quant_rel=None, quant_op=None):
			print('PRECONDITION call returned TRUE')
		else:
			print('PRECONDITION call returned FALSE')
		
		
		
		if pre_post.checkOutliers(belongOp=Belong(0), dataDictionary=data_model_impute_outlier_closest_in, field='Enroll', 
										quant_abs=None, quant_rel=None, quant_op=None):
			print('PRECONDITION call returned TRUE')
		else:
			print('PRECONDITION call returned FALSE')
		
		
		
		if pre_post.checkOutliers(belongOp=Belong(0), dataDictionary=data_model_impute_outlier_closest_in, field='Instate', 
										quant_abs=None, quant_rel=None, quant_op=None):
			print('PRECONDITION call returned TRUE')
		else:
			print('PRECONDITION call returned FALSE')
		
		
		
		#FALTA MIRAR LO DEL BELONG OP DE LAS INVARIANTES
		data_model_impute_outlier_closest_in_copy=invariants.checkInv_SpecialValue_NumOp(dataDictionary=data_model_impute_outlier_closest_in_copy,
														specialTypeInput=SpecialType(2),
														numOpOutput=Operation(3),
														missing_values=None, axis_param=0, field='avg_income')
		print('INVARIANT call')
		
		data_model_impute_outlier_closest_out=data_model_impute_outlier_closest_in_copy.copy()
		data_model_impute_outlier_closest_out.to_csv('../workflow_datasets/data_model_imputeOutliers_out.csv')
		
		
		
		
		
		
		#FALTA MIRAR LO DEL BELONG OP DE LAS INVARIANTES
		data_model_impute_outlier_closest_in_copy=invariants.checkInv_SpecialValue_NumOp(dataDictionary=data_model_impute_outlier_closest_in_copy,
														specialTypeInput=SpecialType(2),
														numOpOutput=Operation(3),
														missing_values=None, axis_param=0, field='distance')
		print('INVARIANT call')
		
		data_model_impute_outlier_closest_out=data_model_impute_outlier_closest_in_copy.copy()
		data_model_impute_outlier_closest_out.to_csv('../workflow_datasets/data_model_imputeOutliers_out.csv')
		
		
		
		
		
		
		#FALTA MIRAR LO DEL BELONG OP DE LAS INVARIANTES
		data_model_impute_outlier_closest_in_copy=invariants.checkInv_SpecialValue_NumOp(dataDictionary=data_model_impute_outlier_closest_in_copy,
														specialTypeInput=SpecialType(2),
														numOpOutput=Operation(3),
														missing_values=None, axis_param=0, field='premiere')
		print('INVARIANT call')
		
		data_model_impute_outlier_closest_out=data_model_impute_outlier_closest_in_copy.copy()
		data_model_impute_outlier_closest_out.to_csv('../workflow_datasets/data_model_imputeOutliers_out.csv')
		
		
		
		
		
		
		#FALTA MIRAR LO DEL BELONG OP DE LAS INVARIANTES
		data_model_impute_outlier_closest_in_copy=invariants.checkInv_SpecialValue_NumOp(dataDictionary=data_model_impute_outlier_closest_in_copy,
														specialTypeInput=SpecialType(2),
														numOpOutput=Operation(3),
														missing_values=None, axis_param=0, field='sex')
		print('INVARIANT call')
		
		data_model_impute_outlier_closest_out=data_model_impute_outlier_closest_in_copy.copy()
		data_model_impute_outlier_closest_out.to_csv('../workflow_datasets/data_model_imputeOutliers_out.csv')
		
		
		
		
		
		
		#FALTA MIRAR LO DEL BELONG OP DE LAS INVARIANTES
		data_model_impute_outlier_closest_in_copy=invariants.checkInv_SpecialValue_NumOp(dataDictionary=data_model_impute_outlier_closest_in_copy,
														specialTypeInput=SpecialType(2),
														numOpOutput=Operation(3),
														missing_values=None, axis_param=0, field='Enroll')
		print('INVARIANT call')
		
		data_model_impute_outlier_closest_out=data_model_impute_outlier_closest_in_copy.copy()
		data_model_impute_outlier_closest_out.to_csv('../workflow_datasets/data_model_imputeOutliers_out.csv')
		
		
		
		
		
		
		#FALTA MIRAR LO DEL BELONG OP DE LAS INVARIANTES
		data_model_impute_outlier_closest_in_copy=invariants.checkInv_SpecialValue_NumOp(dataDictionary=data_model_impute_outlier_closest_in_copy,
														specialTypeInput=SpecialType(2),
														numOpOutput=Operation(3),
														missing_values=None, axis_param=0, field='Instate')
		print('INVARIANT call')
		
		data_model_impute_outlier_closest_out=data_model_impute_outlier_closest_in_copy.copy()
		data_model_impute_outlier_closest_out.to_csv('../workflow_datasets/data_model_imputeOutliers_out.csv')
		
		
		
		
		
		
		if pre_post.checkOutliers(belongOp=Belong(1), dataDictionary=data_model_impute_outlier_closest_out, field='avg_income', 
										quant_abs=None, quant_rel=None, quant_op=None):
			print('POSTCONDITION call returned TRUE')
		else:
			print('POSTCONDITION call returned FALSE')
		
		
		
		if pre_post.checkOutliers(belongOp=Belong(1), dataDictionary=data_model_impute_outlier_closest_out, field='distance', 
										quant_abs=None, quant_rel=None, quant_op=None):
			print('POSTCONDITION call returned TRUE')
		else:
			print('POSTCONDITION call returned FALSE')
		
		
		
		if pre_post.checkOutliers(belongOp=Belong(1), dataDictionary=data_model_impute_outlier_closest_in, field='premiere', 
										quant_abs=None, quant_rel=None, quant_op=None):
			print('POSTCONDITION call returned TRUE')
		else:
			print('POSTCONDITION call returned FALSE')
		
		
		
		if pre_post.checkOutliers(belongOp=Belong(1), dataDictionary=data_model_impute_outlier_closest_out, field='sex', 
										quant_abs=None, quant_rel=None, quant_op=None):
			print('POSTCONDITION call returned TRUE')
		else:
			print('POSTCONDITION call returned FALSE')
		
		
		
		if pre_post.checkOutliers(belongOp=Belong(1), dataDictionary=data_model_impute_outlier_closest_out, field='Enroll', 
										quant_abs=None, quant_rel=None, quant_op=None):
			print('POSTCONDITION call returned TRUE')
		else:
			print('POSTCONDITION call returned FALSE')
		
		
		
		if pre_post.checkOutliers(belongOp=Belong(1), dataDictionary=data_model_impute_outlier_closest_out, field='Instate', 
										quant_abs=None, quant_rel=None, quant_op=None):
			print('POSTCONDITION call returned TRUE')
		else:
			print('POSTCONDITION call returned FALSE')
		
		
#-----------------New DataProcessing-----------------
		data_model_binner_in=pd.read_csv('../workflow_datasets/data_model_stringToNumber_in.csv')
		data_model_binner_in_copy=data_model_binner_in.copy()
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
#-----------------New DataProcessing-----------------
		data_model_binner_in=pd.read_csv('../workflow_datasets/data_model_stringToNumber_in.csv')
		data_model_binner_in_copy=data_model_binner_in.copy()
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
dp=DataProcessing()
dp.generateDataProcessing()
