import pandas as pd
import functions.contract_invariants as contract_invariants
import functions.contract_pre_post as contract_pre_post
from helpers.enumerations import Belong, Operator

class DataProcessing:
	def generateDataProcessing(self):
		pre_post=contract_pre_post.ContractsPrePost()
		invariants=contract_invariants.ContractsInvariants()
#-----------------New DataProcessing-----------------
		data_model_impute_in=pd.read_csv('../data_model.csv')
		missing_values_PRE_value_range_impute_sex_columns=[]
		if pre_post.checkMissingRange(belongOp=Belong(0), dataDictionary=data_model_impute_in, field='sex', 
										missing_values=missing_values_PRE_value_range_impute_sex_columns,
										quant_op=Operator(2), quant_rel=30.0/100):
			print('Precondition call returned TRUE')
		else:
			print('Precondition call returned FALSE')
		
		missing_values_PRE_value_range_impute_IRSCHOOL_columns=[]
		if pre_post.checkMissingRange(belongOp=Belong(0), dataDictionary=data_model_impute_in, field='IRSCHOOL', 
										missing_values=missing_values_PRE_value_range_impute_IRSCHOOL_columns,
										quant_op=Operator(2), quant_rel=30.0/100):
			print('Precondition call returned TRUE')
		else:
			print('Precondition call returned FALSE')
		
		missing_values_PRE_value_range_impute_ETHNICITY_columns=[]
		if pre_post.checkMissingRange(belongOp=Belong(0), dataDictionary=data_model_impute_in, field='ETHNICITY', 
										missing_values=missing_values_PRE_value_range_impute_ETHNICITY_columns,
										quant_op=Operator(2), quant_rel=30.0/100):
			print('Precondition call returned TRUE')
		else:
			print('Precondition call returned FALSE')
		
		
		
		
		
		
		
#-----------------New DataProcessing-----------------
		data_model_impute_sex_in=pd.read_csv('../data_model.csv')
		missing_values_PRE_value_range_impute_sex=[]
		if pre_post.checkMissingRange(belongOp=Belong(0), dataDictionary=data_model_impute_sex_in, field='sex', 
										missing_values=missing_values_PRE_value_range_impute_sex,
										quant_op=Operator(2), quant_rel=70.0/100):
			print('Precondition call returned TRUE')
		else:
			print('Precondition call returned FALSE')
		
		
		
#-----------------New DataProcessing-----------------
		data_model_impute_IRSCHOOL_in=pd.read_csv('../data_model.csv')
		missing_values_PRE_value_range_impute_IRSCHOOL=[]
		if pre_post.checkMissingRange(belongOp=Belong(0), dataDictionary=data_model_impute_IRSCHOOL_in, field='IRSCHOOL', 
										missing_values=missing_values_PRE_value_range_impute_IRSCHOOL,
										quant_op=Operator(2), quant_rel=30.0/100):
			print('Precondition call returned TRUE')
		else:
			print('Precondition call returned FALSE')
		
		
		
#-----------------New DataProcessing-----------------
		data_model_impute_ETHNICITY_in=pd.read_csv('../data_model.csv')
		missing_values_PRE_value_range_impute_ETHNICITY=[]
		if pre_post.checkMissingRange(belongOp=Belong(0), dataDictionary=data_model_impute_ETHNICITY_in, field='ETHNICITY', 
										missing_values=missing_values_PRE_value_range_impute_ETHNICITY,
										quant_op=Operator(2), quant_rel=30.0/100):
			print('Precondition call returned TRUE')
		else:
			print('Precondition call returned FALSE')
		
		
		
#-----------------New DataProcessing-----------------
		data_model_impute_ACADEMIC_INTEREST_2_in=pd.read_csv('../data_model.csv')
		missing_values_PRE_value_range_impute_ACADEMIC_INTEREST_2=[]
		if pre_post.checkMissingRange(belongOp=Belong(0), dataDictionary=data_model_impute_ACADEMIC_INTEREST_2_in, field='ACADEMIC_INTEREST_2', 
										missing_values=missing_values_PRE_value_range_impute_ACADEMIC_INTEREST_2,
										quant_op=Operator(2), quant_rel=30.0/100):
			print('Precondition call returned TRUE')
		else:
			print('Precondition call returned FALSE')
		
		
		
		missing_values_PRE_value_range_impute_ACADEMIC_INTEREST_1=[]
		if pre_post.checkMissingRange(belongOp=Belong(0), dataDictionary=data_model_impute_ACADEMIC_INTEREST_2_in, field='ACADEMIC_INTEREST_1', 
										missing_values=missing_values_PRE_value_range_impute_ACADEMIC_INTEREST_1,
										quant_op=Operator(2), quant_rel=30.0/100):
			print('Precondition call returned TRUE')
		else:
			print('Precondition call returned FALSE')
		
		
		
#-----------------New DataProcessing-----------------
		data_model_impute_mean_in=pd.read_csv('../data_model.csv')
		
		
		
		
		
		
#-----------------New DataProcessing-----------------
		data_model_impute_linear_interpolation_in=pd.read_csv('../data_model.csv')
		
		
#-----------------New DataProcessing-----------------
#-----------------New DataProcessing-----------------
#-----------------New DataProcessing-----------------
#-----------------New DataProcessing-----------------
#-----------------New DataProcessing-----------------
		
		
#-----------------New DataProcessing-----------------
		
		
		
		
		
#-----------------New DataProcessing-----------------
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
#-----------------New DataProcessing-----------------
		
		
		
		
		
		
		
		
		
#-----------------New DataProcessing-----------------
		
		
dp=DataProcessing()
dp.generateDataProcessing()
