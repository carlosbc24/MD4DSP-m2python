import pandas as pd
import functions.contract_invariants as contract_invariants
import functions.contract_pre_post as contract_pre_post
import functions.data_transformations as data_transformations
from helpers.enumerations import Belong, Operator, Operation, SpecialType, DataType, DerivedType, Closure, FilterType
from helpers.logger import set_logger

def generateDataProcessing():
#-----------------New DataProcessing-----------------
	imputeByDerivedValue_input_dataDictionary=pd.read_csv('./knime_dataDictionaries/missing_input_dataDictionary.csv', sep = ',')
	missing_values_imputeByDerivedValue_PRE_valueRange=[]
	if contract_pre_post.check_missing_range(belong_op=Belong(0), data_dictionary=imputeByDerivedValue_input_dataDictionary, field='sex', 
									missing_values=missing_values_imputeByDerivedValue_PRE_valueRange,
									quant_op=Operator(2), quant_rel=30.0/100):
		print('PRECONDITION imputeMissingByMostFrequent(sex)_PRE_value_range VALIDATED')
	else:
		print('PRECONDITION imputeMissingByMostFrequent(sex)_PRE_value_range NOT VALIDATED')
	
	missing_values_imputeByDerivedValue_PRE_valueRange=[]
	if contract_pre_post.check_missing_range(belong_op=Belong(0), data_dictionary=imputeByDerivedValue_input_dataDictionary, field='IRSCHOOL', 
									missing_values=missing_values_imputeByDerivedValue_PRE_valueRange,
									quant_op=Operator(2), quant_rel=30.0/100):
		print('PRECONDITION imputeMissingByMostFrequent(IRSCHOOL)_PRE_valueRange VALIDATED')
	else:
		print('PRECONDITION imputeMissingByMostFrequent(IRSCHOOL)_PRE_valueRange NOT VALIDATED')
	
	missing_values_imputeByDerivedValue_PRE_valueRange=[]
	if contract_pre_post.check_missing_range(belong_op=Belong(0), data_dictionary=imputeByDerivedValue_input_dataDictionary, field='ETHNICITY', 
									missing_values=missing_values_imputeByDerivedValue_PRE_valueRange,
									quant_op=Operator(2), quant_rel=30.0/100):
		print('PRECONDITION imputeMissingByMostFrequent(ETHNICITY)_PRE_valueRange VALIDATED')
	else:
		print('PRECONDITION imputeMissingByMostFrequent(ETHNICITY)_PRE_valueRange NOT VALIDATED')
	
	imputeByDerivedValue_input_dataDictionary_transformed=imputeByDerivedValue_input_dataDictionary.copy()
	missing_values_list=[]
	
	imputeByDerivedValue_input_dataDictionary_transformed=data_transformations.transform_special_value_derived_value(data_dictionary=imputeByDerivedValue_input_dataDictionary_transformed,
																  special_type_input=SpecialType(0), derived_type_output=DerivedType(0),
																  missing_values=missing_values_list,		
																  axis_param=0, field_in = 'sex', field_out = 'sex')
	
	missing_values_list=[]
	
	imputeByDerivedValue_input_dataDictionary_transformed=data_transformations.transform_special_value_derived_value(data_dictionary=imputeByDerivedValue_input_dataDictionary_transformed,
																  special_type_input=SpecialType(0), derived_type_output=DerivedType(0),
																  missing_values=missing_values_list,		
																  axis_param=0, field_in = 'IRSCHOOL', field_out = 'IRSCHOOL')
	
	missing_values_list=[]
	
	imputeByDerivedValue_input_dataDictionary_transformed=data_transformations.transform_special_value_derived_value(data_dictionary=imputeByDerivedValue_input_dataDictionary_transformed,
																  special_type_input=SpecialType(0), derived_type_output=DerivedType(0),
																  missing_values=missing_values_list,		
																  axis_param=0, field_in = 'ETHNICITY', field_out = 'ETHNICITY')
	
	imputeByDerivedValue_output_dataDictionary=imputeByDerivedValue_input_dataDictionary_transformed
	imputeByDerivedValue_output_dataDictionary.to_csv('./knime_dataDictionaries/imputeMissingByMostFrequent(sex, IRISCHOOL, ETHNICITY)_output_dataDictionary.csv')
	
	missing_values_imputeByDerivedValue_POST_valueRange=[]
	if contract_pre_post.check_missing_range(belong_op=Belong(1), data_dictionary=imputeByDerivedValue_output_dataDictionary, field='sex', 
									missing_values=missing_values_imputeByDerivedValue_POST_valueRange,
									quant_abs=None, quant_rel=None, quant_op=None):
		print('POSTCONDITION POST_value_range_impute_sex_columns VALIDATED')
	else:
		print('POSTCONDITION POST_value_range_impute_sex_columns NOT VALIDATED')
	
	missing_values_imputeByDerivedValue_POST_valueRange=[]
	if contract_pre_post.check_missing_range(belong_op=Belong(1), data_dictionary=imputeByDerivedValue_output_dataDictionary, field='IRSCHOOL', 
									missing_values=missing_values_imputeByDerivedValue_POST_valueRange,
									quant_abs=None, quant_rel=None, quant_op=None):
		print('POSTCONDITION POST_value_range_impute_IRSCHOOL_columns VALIDATED')
	else:
		print('POSTCONDITION POST_value_range_impute_IRSCHOOL_columns NOT VALIDATED')
	
	missing_values_imputeByDerivedValue_POST_valueRange=[]
	if contract_pre_post.check_missing_range(belong_op=Belong(1), data_dictionary=imputeByDerivedValue_output_dataDictionary, field='ETHNICITY', 
									missing_values=missing_values_imputeByDerivedValue_POST_valueRange,
									quant_abs=None, quant_rel=None, quant_op=None):
		print('POSTCONDITION POST_value_range_impute_ETHNICITY_columns VALIDATED')
	else:
		print('POSTCONDITION POST_value_range_impute_ETHNICITY_columns NOT VALIDATED')
	
	missing_values_imputeByDerivedValue_INV_condition=[]
	if contract_invariants.check_inv_special_value_derived_value(data_dictionary_in=imputeByDerivedValue_input_dataDictionary,
								data_dictionary_out=imputeByDerivedValue_output_dataDictionary,
								belong_op_in=Belong(0),
								belong_op_out=Belong(0),
								special_type_input=SpecialType(0),
								derived_type_output=DerivedType(0),
								missing_values=missing_values_imputeByDerivedValue_INV_condition, axis_param=0, field_in='sex', field_out='sex'):
		print('INVARIANT INV_condition_impute_sex_columns VALIDATED')
	else:
		print('INVARIANT INV_condition_impute_sex_columns NOT VALIDATED')
	
	
	missing_values_imputeByDerivedValue_INV_condition=[]
	if contract_invariants.check_inv_special_value_derived_value(data_dictionary_in=imputeByDerivedValue_input_dataDictionary,
								data_dictionary_out=imputeByDerivedValue_output_dataDictionary,
								belong_op_in=Belong(0),
								belong_op_out=Belong(0),
								special_type_input=SpecialType(0),
								derived_type_output=DerivedType(0),
								missing_values=missing_values_imputeByDerivedValue_INV_condition, axis_param=0, field_in='IRSCHOOL', field_out='IRSCHOOL'):
		print('INVARIANT INV_condition_impute_IRSCHOOL_columns VALIDATED')
	else:
		print('INVARIANT INV_condition_impute_IRSCHOOL_columns NOT VALIDATED')
	
	
	missing_values_imputeByDerivedValue_INV_condition=[]
	if contract_invariants.check_inv_special_value_derived_value(data_dictionary_in=imputeByDerivedValue_input_dataDictionary,
								data_dictionary_out=imputeByDerivedValue_output_dataDictionary,
								belong_op_in=Belong(0),
								belong_op_out=Belong(0),
								special_type_input=SpecialType(0),
								derived_type_output=DerivedType(0),
								missing_values=missing_values_imputeByDerivedValue_INV_condition, axis_param=0, field_in='ETHNICITY', field_out='ETHNICITY'):
		print('INVARIANT INV_condition_impute_ETHNICITY_columns VALIDATED')
	else:
		print('INVARIANT INV_condition_impute_ETHNICITY_columns NOT VALIDATED')
	
	
#-----------------New DataProcessing-----------------
	imputeByFixValue_input_dataDictionary=pd.read_csv('./knime_dataDictionaries/imputeMissingByMostFrequent(sex, IRISCHOOL, ETHNICITY)_output_dataDictionary.csv', sep=',')

	missing_values_imputeByFixValue_PRE_valueRange=[]
	if contract_pre_post.check_missing_range(belong_op=Belong(0), data_dictionary=imputeByFixValue_input_dataDictionary, field='ACADEMIC_INTEREST_2', 
									missing_values=missing_values_imputeByFixValue_PRE_valueRange,
									quant_op=Operator(2), quant_rel=30.0/100):
		print('PRECONDITION imputeMissingByFixValue(ACADEMIC_INTEREST_2)_PRE_valueRange VALIDATED')
	else:
		print('PRECONDITION imputeMissingByFixValue(ACADEMIC_INTEREST_2)_PRE_valueRange NOT VALIDATED')
	
	missing_values_imputeByFixValue_PRE_valueRange=[]
	if contract_pre_post.check_missing_range(belong_op=Belong(0), data_dictionary=imputeByFixValue_input_dataDictionary, field='ACADEMIC_INTEREST_1', 
									missing_values=missing_values_imputeByFixValue_PRE_valueRange,
									quant_op=Operator(2), quant_rel=30.0/100):
		print('PRECONDITION imputeMissingByFixValue(ACADEMIC_INTEREST_1)_PRE_valueRange VALIDATED')
	else:
		print('PRECONDITION imputeMissingByFixValue(ACADEMIC_INTEREST_1)_PRE_valueRange NOT VALIDATED')
	
	imputeByFixValue_input_dataDictionary_transformed=imputeByFixValue_input_dataDictionary.copy()
	missing_values_list=[]
	
	imputeByFixValue_input_dataDictionary_transformed=data_transformations.transform_special_value_fix_value(data_dictionary=imputeByFixValue_input_dataDictionary_transformed,
																  special_type_input=SpecialType(0), fix_value_output='Unknown',
																  missing_values=missing_values_list,		
							                                      data_type_output = DataType(0),
																  axis_param=0, field_in = 'ACADEMIC_INTEREST_2', field_out = 'ACADEMIC_INTEREST_2')
	
	missing_values_list=[]
	
	imputeByFixValue_input_dataDictionary_transformed=data_transformations.transform_special_value_fix_value(data_dictionary=imputeByFixValue_input_dataDictionary_transformed,
																  special_type_input=SpecialType(0), fix_value_output='Unknown',
																  missing_values=missing_values_list,		
							                                      data_type_output = DataType(0),
																  axis_param=0, field_in = 'ACADEMIC_INTEREST_1', field_out = 'ACADEMIC_INTEREST_1')
	
	imputeByFixValue_output_dataDictionary=imputeByFixValue_input_dataDictionary_transformed
	imputeByFixValue_output_dataDictionary.to_csv('./knime_dataDictionaries/imputeMissingByFixValue(ACADEMIC_INTEREST_2, ACADEMIC_INTEREST_1)_output_dataDictionary.csv')
	
	missing_values_imputeByFixValue_POST_valueRange=[]
	if contract_pre_post.check_missing_range(belong_op=Belong(1), data_dictionary=imputeByFixValue_output_dataDictionary, field='ACADEMIC_INTEREST_2', 
									missing_values=missing_values_imputeByFixValue_POST_valueRange,
									quant_abs=None, quant_rel=None, quant_op=None):
		print('POSTCONDITION imputeMissingByFixValue(ACADEMIC_INTEREST_2)_POST_valueRange VALIDATED')
	else:
		print('POSTCONDITION imputeMissingByFixValue(ACADEMIC_INTEREST_2)_POST_valueRange NOT VALIDATED')
	
	missing_values_imputeByFixValue_POST_valueRange=[]
	if contract_pre_post.check_missing_range(belong_op=Belong(1), data_dictionary=imputeByFixValue_output_dataDictionary, field='ACADEMIC_INTEREST_1', 
									missing_values=missing_values_imputeByFixValue_POST_valueRange,
									quant_abs=None, quant_rel=None, quant_op=None):
		print('POSTCONDITION imputeMissingByFixValue(ACADEMIC_INTEREST_1)_POST_valueRange VALIDATED')
	else:
		print('POSTCONDITION imputeMissingByFixValue(ACADEMIC_INTEREST_1)_POST_valueRange NOT VALIDATED')
	
	missing_values_imputeByFixValue_INV_condition=[]
	if contract_invariants.check_inv_special_value_fix_value(data_dictionary_in=imputeByFixValue_input_dataDictionary,
								data_dictionary_out=imputeByFixValue_output_dataDictionary,
								special_type_input=SpecialType(0),
								fix_value_output='Unknown',
								belong_op_in=Belong(0),
								belong_op_out=Belong(0),
								data_type_output=DataType(0),
								missing_values=missing_values_imputeByFixValue_INV_condition, 
								axis_param=0, field_in='ACADEMIC_INTEREST_2', field_out='ACADEMIC_INTEREST_2'):
		print('INVARIANT imputeMissingByFixValue(ACADEMIC_INTEREST_2)_INV_condition VALIDATED')
	else:
		print('INVARIANT imputeMissingByFixValue(ACADEMIC_INTEREST_2)_INV_condition NOT VALIDATED')
	
	
	missing_values_imputeByFixValue_INV_condition=[]
	if contract_invariants.check_inv_special_value_fix_value(data_dictionary_in=imputeByFixValue_input_dataDictionary,
								data_dictionary_out=imputeByFixValue_output_dataDictionary,
								special_type_input=SpecialType(0),
								fix_value_output='Unknown',
								belong_op_in=Belong(0),
								belong_op_out=Belong(0),
								data_type_output=DataType(0),
								missing_values=missing_values_imputeByFixValue_INV_condition, 
								axis_param=0, field_in='ACADEMIC_INTEREST_1', field_out='ACADEMIC_INTEREST_1'):
		print('INVARIANT imputeMissingByFixValue(ACADEMIC_INTEREST_1)_INV_condition VALIDATED')
	else:
		print('INVARIANT imputeMissingByFixValue(ACADEMIC_INTEREST_1)_INV_condition NOT VALIDATED')
	
	
#-----------------New DataProcessing-----------------
	imputeByNumericOp_input_dataDictionary=pd.read_csv('./knime_dataDictionaries/imputeMissingByFixValue(ACADEMIC_INTEREST_2, ACADEMIC_INTEREST_1)_output_dataDictionary.csv', sep=',')

	missing_values_imputeByNumericOp_PRE_valueRange=[]
	if contract_pre_post.check_missing_range(belong_op=Belong(0), data_dictionary=imputeByNumericOp_input_dataDictionary, field='avg_income', 
									missing_values=missing_values_imputeByNumericOp_PRE_valueRange,
									quant_abs=None, quant_rel=None, quant_op=None):
		print('PRECONDITION imputeMissingByMean(avg_income)_PRE_valueRange VALIDATED')
	else:
		print('PRECONDITION imputeMissingByMean(avg_income)_PRE_valueRange NOT VALIDATED')
	
	missing_values_imputeByNumericOp_PRE_valueRange=[]
	if contract_pre_post.check_missing_range(belong_op=Belong(0), data_dictionary=imputeByNumericOp_input_dataDictionary, field='distance', 
									missing_values=missing_values_imputeByNumericOp_PRE_valueRange,
									quant_abs=None, quant_rel=None, quant_op=None):
		print('PRECONDITION imputeMissingByMean(distance)_PRE_valueRange VALIDATED')
	else:
		print('PRECONDITION imputeMissingByMean(distance)_PRE_valueRange NOT VALIDATED')
	
	imputeByNumericOp_input_dataDictionary_transformed=imputeByNumericOp_input_dataDictionary.copy()
	missing_values_list=[]
	
	imputeByNumericOp_input_dataDictionary_transformed=data_transformations.transform_special_value_num_op(data_dictionary=imputeByNumericOp_input_dataDictionary_transformed,
																  special_type_input=SpecialType(0), num_op_output=Operation(1),
																  missing_values=missing_values_list,		
																  axis_param=0, field_in = 'avg_income', field_out = 'avg_income')
	
	missing_values_list=[]
	
	imputeByNumericOp_input_dataDictionary_transformed=data_transformations.transform_special_value_num_op(data_dictionary=imputeByNumericOp_input_dataDictionary_transformed,
																  special_type_input=SpecialType(0), num_op_output=Operation(1),
																  missing_values=missing_values_list,		
																  axis_param=0, field_in = 'distance', field_out = 'distance')
	
	imputeByNumericOp_output_dataDictionary=imputeByNumericOp_input_dataDictionary_transformed
	imputeByNumericOp_output_dataDictionary.to_csv('./knime_dataDictionaries/imputeMissingByMean(avg_income, distance)_output_dataDictionary.csv')
	
	missing_values_imputeByNumericOp_POST_valueRange=[]
	if contract_pre_post.check_missing_range(belong_op=Belong(1), data_dictionary=imputeByNumericOp_output_dataDictionary, field='avg_income', 
									missing_values=missing_values_imputeByNumericOp_POST_valueRange,
									quant_abs=None, quant_rel=None, quant_op=None):
		print('POSTCONDITION imputeMissingByMean(avg_income)_POST_valueRange VALIDATED')
	else:
		print('POSTCONDITION imputeMissingByMean(avg_income)_POST_valueRange NOT VALIDATED')
	
	missing_values_imputeByNumericOp_POST_valueRange=[]
	if contract_pre_post.check_missing_range(belong_op=Belong(1), data_dictionary=imputeByNumericOp_output_dataDictionary, field='distance', 
									missing_values=missing_values_imputeByNumericOp_POST_valueRange,
									quant_abs=None, quant_rel=None, quant_op=None):
		print('POSTCONDITION imputeMissingByMean(distance)_POST_valueRange VALIDATED')
	else:
		print('POSTCONDITION imputeMissingByMean(distance)_POST_valueRange NOT VALIDATED')
	
	missing_values_imputeByNumericOp_INV_condition=[]
	if contract_invariants.check_inv_special_value_num_op(data_dictionary_in=imputeByNumericOp_input_dataDictionary,
											data_dictionary_out=imputeByNumericOp_output_dataDictionary,
											belong_op_in=Belong(0),
											belong_op_out=Belong(0),
											special_type_input=SpecialType(0),
											num_op_output=Operation(1),
											missing_values=missing_values_imputeByNumericOp_INV_condition, axis_param=0, field_in='avg_income', field_out='avg_income'):
		print('INVARIANT imputeMissingByMean(avg_income)_INV_condition VALIDATED')
	else:
		print('INVARIANT imputeMissingByMean(avg_income)_INV_condition NOT VALIDATED')
	
	
	missing_values_imputeByNumericOp_INV_condition=[]
	if contract_invariants.check_inv_special_value_num_op(data_dictionary_in=imputeByNumericOp_input_dataDictionary,
											data_dictionary_out=imputeByNumericOp_output_dataDictionary,
											belong_op_in=Belong(0),
											belong_op_out=Belong(0),
											special_type_input=SpecialType(0),
											num_op_output=Operation(1),
											missing_values=missing_values_imputeByNumericOp_INV_condition, axis_param=0, field_in='distance', field_out='distance'):
		print('INVARIANT imputeMissingByMean(distance)_INV_condition VALIDATED')
	else:
		print('INVARIANT imputeMissingByMean(distance)_INV_condition NOT VALIDATED')
	
	
#-----------------New DataProcessing-----------------
	imputeByNumericOp_input_dataDictionary=pd.read_csv('./knime_dataDictionaries/imputeMissingByMean(avg_income, distance)_output_dataDictionary.csv', sep=',')

	missing_values_imputeByNumericOp_PRE_valueRange=[]
	if contract_pre_post.check_missing_range(belong_op=Belong(0), data_dictionary=imputeByNumericOp_input_dataDictionary, field='satscore', 
									missing_values=missing_values_imputeByNumericOp_PRE_valueRange,
									quant_op=Operator(3), quant_rel=60.0/100):
		print('PRECONDITION imputeMissingByLinearInterpolation(satscore)_PRE_valueRange VALIDATED')
	else:
		print('PRECONDITION imputeMissingByLinearInterpolation(satscore)_PRE_valueRange NOT VALIDATED')
	
	imputeByNumericOp_input_dataDictionary_transformed=imputeByNumericOp_input_dataDictionary.copy()
	missing_values_list=[]
	
	imputeByNumericOp_input_dataDictionary_transformed=data_transformations.transform_special_value_num_op(data_dictionary=imputeByNumericOp_input_dataDictionary_transformed,
																  special_type_input=SpecialType(0), num_op_output=Operation(0),
																  missing_values=missing_values_list,		
																  axis_param=0, field_in = 'satscore', field_out = 'satscore')
	
	imputeByNumericOp_output_dataDictionary=imputeByNumericOp_input_dataDictionary_transformed
	imputeByNumericOp_output_dataDictionary.to_csv('./knime_dataDictionaries/imputeMissingByLinearInterpolation(satscore)_output_dataDictionary.csv')
	
	missing_values_imputeByNumericOp_POST_valueRange=[]
	if contract_pre_post.check_missing_range(belong_op=Belong(1), data_dictionary=imputeByNumericOp_output_dataDictionary, field='satscore', 
									missing_values=missing_values_imputeByNumericOp_POST_valueRange,
									quant_abs=None, quant_rel=None, quant_op=None):
		print('POSTCONDITION imputeMissingByLinearInterpolation(satscore)_POST_valueRange VALIDATED')
	else:
		print('POSTCONDITION imputeMissingByLinearInterpolation(satscore)_POST_valueRange NOT VALIDATED')
	
	missing_values_imputeByNumericOp_INV_condition=[]
	if contract_invariants.check_inv_special_value_num_op(data_dictionary_in=imputeByNumericOp_input_dataDictionary,
											data_dictionary_out=imputeByNumericOp_output_dataDictionary,
											belong_op_in=Belong(0),
											belong_op_out=Belong(0),
											special_type_input=SpecialType(0),
											num_op_output=Operation(0),
											missing_values=missing_values_imputeByNumericOp_INV_condition, axis_param=0, field_in='satscore', field_out='satscore'):
		print('INVARIANT imputeMissingByLinearInterpolation(satscore)_INV_condition VALIDATED')
	else:
		print('INVARIANT imputeMissingByLinearInterpolation(satscore)_INV_condition NOT VALIDATED')
	
	
#-----------------New DataProcessing-----------------
	rowFilter_input_DataDictionary=pd.read_csv('./knime_dataDictionaries/imputeMissingByLinearInterpolation(satscore)_output_dataDictionary.csv', sep=',')

	if contract_pre_post.check_fix_value_range(value=0, data_dictionary=rowFilter_input_DataDictionary, belong_op=Belong(0), field='init_span',
									quant_abs=None, quant_rel=None, quant_op=None):
		print('PRECONDITION rowFilter(init_span)_PRE_valueRange VALIDATED')
	else:
		print('PRECONDITION rowFilter(init_span)_PRE_valueRange NOT VALIDATED')
	
	rowFilter_input_DataDictionary_transformed=rowFilter_input_DataDictionary.copy()
	columns_rowFilter_param_filter=['init_span']
	
	filter_fix_value_list_rowFilter_param_filter=[0]
	
	rowFilter_input_DataDictionary_transformed=data_transformations.transform_filter_rows_primitive(data_dictionary=rowFilter_input_DataDictionary_transformed,
																											columns=columns_rowFilter_param_filter,
																		                                    filter_fix_value_list=filter_fix_value_list_rowFilter_param_filter,
																											filter_type=FilterType(0))
	rowFilter_output_DataDictionary=rowFilter_input_DataDictionary_transformed
	rowFilter_output_DataDictionary.to_csv('./knime_dataDictionaries/rowFilter_output_dataDictionary.csv')
	
	if contract_pre_post.check_fix_value_range(value='0', data_dictionary=rowFilter_output_DataDictionary, belong_op=Belong(1), field='init_span',
									quant_abs=None, quant_rel=None, quant_op=None):
		print('POSTCONDITION rowFilter(init_span)_POST_valueRange VALIDATED')
	else:
		print('POSTCONDITION rowFilter(init_span)_POST_valueRange NOT VALIDATED')
	
#-----------------New DataProcessing-----------------
	columnFilter_input_DataDictionary=pd.read_csv('./knime_dataDictionaries/rowFilter_output_dataDictionary.csv', sep=',')

	field_list_columnFilter_PRE_field_range=['TRAVEL_INIT_CNTCTS', 'REFERRAL_CNCTS', 'telecq', 'stuemail', 'interest']
	if contract_pre_post.check_field_range(fields=field_list_columnFilter_PRE_field_range,
								data_dictionary=columnFilter_input_DataDictionary,
								belong_op=Belong(0)):
		print('PRECONDITION columnFilter(TRAVEL_INIT_CNTCTS, REFERRAL_CNCTS, telecq, interest, stuemail, CONTACT_CODE1)_PRE_fieldRange VALIDATED')
	else:
		print('PRECONDITION columnFilter(TRAVEL_INIT_CNTCTS, REFERRAL_CNCTS, telecq, interest, stuemail, CONTACT_CODE1)_PRE_fieldRange NOT VALIDATED')
	
	
	columnFilter_input_DataDictionary_transformed=columnFilter_input_DataDictionary.copy()
	field_list_columnFilter_param_field=['TRAVEL_INIT_CNTCTS', 'REFERRAL_CNCTS']
	
	columnFilter_input_DataDictionary_transformed=data_transformations.transform_filter_columns(data_dictionary=columnFilter_input_DataDictionary_transformed,
																	columns=field_list_columnFilter_param_field, belong_op=Belong.BELONG)
	
	columnFilter_output_DataDictionary=columnFilter_input_DataDictionary_transformed
	columnFilter_output_DataDictionary.to_csv('./knime_dataDictionaries/columnFilter_output_dataDictionary.csv')
	
	field_list_columnFilter_POST_field_range=['stuemail', 'interest', 'telecq', 'TRAVEL_INIT_CNTCTS', 'REFERRAL_CNCTS']
	if contract_pre_post.check_field_range(fields=field_list_columnFilter_POST_field_range,
								data_dictionary=columnFilter_output_DataDictionary,
								belong_op=Belong(1)):
		print('POSTCONDITION columnFilter(TRAVEL_INIT_CNTCTS, REFERRAL_CNCTS, telecq, interest, stuemail, CONTACT_CODE1)_POST_fieldRange VALIDATED')
	else:
		print('POSTCONDITION columnFilter(TRAVEL_INIT_CNTCTS, REFERRAL_CNCTS, telecq, interest, stuemail, CONTACT_CODE1)_POST_fieldRange NOT VALIDATED')
	
	
#-----------------New DataProcessing-----------------
	mapping_input_dataDictionary=pd.read_csv('./knime_dataDictionaries/columnFilter_output_dataDictionary.csv', sep=',')

	if contract_pre_post.check_fix_value_range(value='A', data_dictionary=mapping_input_dataDictionary, belong_op=Belong(0), field='TERRITORY',
									quant_abs=None, quant_rel=None, quant_op=None):
		print('PRECONDITION mapping(TERRITORY)_PRE_valueRange VALIDATED')
	else:
		print('PRECONDITION mapping(TERRITORY)_PRE_valueRange NOT VALIDATED')
	if contract_pre_post.check_fix_value_range(value='N', data_dictionary=mapping_input_dataDictionary, belong_op=Belong(0), field='TERRITORY',
									quant_abs=None, quant_rel=None, quant_op=None):
		print('PRECONDITION mapping(TERRITORY)_PRE_valueRange VALIDATED')
	else:
		print('PRECONDITION mapping(TERRITORY)_PRE_valueRange NOT VALIDATED')
	
	input_values_list=['A', 'N']
	output_values_list=['0', '0']
	data_type_input_list=[DataType(0), DataType(0)]
	data_type_output_list=[DataType(0), DataType(0)]
	
	
	mapping_output_dataDictionary=data_transformations.transform_fix_value_fix_value(data_dictionary=mapping_input_dataDictionary, input_values_list=input_values_list,
																  output_values_list=output_values_list,
							                                      data_type_input_list = data_type_input_list,
							                                      data_type_output_list = data_type_output_list, field_in = 'TERRITORY', field_out = 'TERRITORY')
	
	mapping_output_dataDictionary.to_csv('./knime_dataDictionaries/ruleEngine_territory_output_dataDictionary.csv')
	
	if contract_pre_post.check_fix_value_range(value='A', data_dictionary=mapping_output_dataDictionary, belong_op=Belong(0), field='TERRITORY',
									quant_abs=None, quant_rel=None, quant_op=None):
		print('POSTCONDITION mapping(TERRITORY)_POST_valueRange VALIDATED')
	else:
		print('POSTCONDITION mapping(TERRITORY)_POST_valueRange NOT VALIDATED')
	if contract_pre_post.check_fix_value_range(value='N', data_dictionary=mapping_output_dataDictionary, belong_op=Belong(0), field='TERRITORY',
									quant_abs=None, quant_rel=None, quant_op=None):
		print('POSTCONDITION mapping(TERRITORY)_POST_valueRange VALIDATED')
	else:
		print('POSTCONDITION mapping(TERRITORY)_POST_valueRange NOT VALIDATED')
	
	
	input_values_list_def_INV_condition=['A', 'N']
	output_values_list_def_INV_condition=['0', '0']
	
	data_type_input_list_def_INV_condition=[DataType(0), DataType(0)]
	data_type_output_list_def_INV_condition=[DataType(0), DataType(0)]
	
	if contract_invariants.check_inv_fix_value_fix_value(data_dictionary_in=mapping_input_dataDictionary,
											data_dictionary_out=mapping_output_dataDictionary,
											input_values_list=input_values_list_def_INV_condition, 
											output_values_list=output_values_list_def_INV_condition,
											belong_op_in=Belong(0),
											belong_op_out=Belong(0),
											data_type_input_list=data_type_input_list_def_INV_condition,
											data_type_output_list=data_type_output_list_def_INV_condition,
											field_in='TERRITORY', field_out='TERRITORY'):
		print('INVARIANT Mapping(TERRITORY)_INV_condition VALIDATED')
	else:
		print('INVARIANT Mapping(TERRITORY)_INV_condition NOT VALIDATED')
	
	
#-----------------New DataProcessing-----------------
	mapping_input_dataDictionary=pd.read_csv('./knime_dataDictionaries/ruleEngine_territory_output_dataDictionary.csv', sep=',')

	if contract_pre_post.check_fix_value_range(value='Y', data_dictionary=mapping_input_dataDictionary, belong_op=Belong(0), field='Instate',
									quant_abs=None, quant_rel=None, quant_op=None):
		print('PRECONDITION mapping(Instate)_PRE_valueRange VALIDATED')
	else:
		print('PRECONDITION mapping(Instate)_PRE_valueRange NOT VALIDATED')
	if contract_pre_post.check_fix_value_range(value='N', data_dictionary=mapping_input_dataDictionary, belong_op=Belong(0), field='Instate',
									quant_abs=None, quant_rel=None, quant_op=None):
		print('PRECONDITION mapping(Instate)_PRE_valueRange VALIDATED')
	else:
		print('PRECONDITION mapping(Instate)_PRE_valueRange NOT VALIDATED')
	
	input_values_list=['Y', 'N']
	output_values_list=['1', '0']
	data_type_input_list=[DataType(0), DataType(0)]
	data_type_output_list=[DataType(0), DataType(0)]
	
	
	mapping_output_dataDictionary=data_transformations.transform_fix_value_fix_value(data_dictionary=mapping_input_dataDictionary, input_values_list=input_values_list,
																  output_values_list=output_values_list,
							                                      data_type_input_list = data_type_input_list,
							                                      data_type_output_list = data_type_output_list, field_in = 'Instate', field_out = 'Instate')
	
	mapping_output_dataDictionary.to_csv('./knime_dataDictionaries/ruleEngine_instate_output_dataDictionary.csv')
	
	if contract_pre_post.check_fix_value_range(value='Y', data_dictionary=mapping_output_dataDictionary, belong_op=Belong(0), field='Instate',
									quant_abs=None, quant_rel=None, quant_op=None):
		print('POSTCONDITION mapping(Instate)_POST_valueRange VALIDATED')
	else:
		print('POSTCONDITION mapping(Instate)_POST_valueRange NOT VALIDATED')
	if contract_pre_post.check_fix_value_range(value='N', data_dictionary=mapping_output_dataDictionary, belong_op=Belong(0), field='Instate',
									quant_abs=None, quant_rel=None, quant_op=None):
		print('POSTCONDITION mapping(Instate)_POST_valueRange VALIDATED')
	else:
		print('POSTCONDITION mapping(Instate)_POST_valueRange NOT VALIDATED')
	
	
	input_values_list_def_INV_condition=['Y', 'N']
	output_values_list_def_INV_condition=['1', '0']
	
	data_type_input_list_def_INV_condition=[DataType(0), DataType(0)]
	data_type_output_list_def_INV_condition=[DataType(0), DataType(0)]
	
	if contract_invariants.check_inv_fix_value_fix_value(data_dictionary_in=mapping_input_dataDictionary,
											data_dictionary_out=mapping_output_dataDictionary,
											input_values_list=input_values_list_def_INV_condition, 
											output_values_list=output_values_list_def_INV_condition,
											belong_op_in=Belong(0),
											belong_op_out=Belong(0),
											data_type_input_list=data_type_input_list_def_INV_condition,
											data_type_output_list=data_type_output_list_def_INV_condition,
											field_in='Instate', field_out='Instate'):
		print('INVARIANT mapping(Instate)_INV_condition VALIDATED')
	else:
		print('INVARIANT mapping(Instate)_INV_condition NOT VALIDATED')
	
	
#-----------------New DataProcessing-----------------
	categoricalToContinuous_input_dataDictionary=pd.read_csv('./knime_dataDictionaries/ruleEngine_instate_output_dataDictionary.csv', sep=',')

	categoricalToContinuous_input_dataDictionary_transformed=categoricalToContinuous_input_dataDictionary.copy()
	categoricalToContinuous_input_dataDictionary_transformed=data_transformations.transform_cast_type(data_dictionary=categoricalToContinuous_input_dataDictionary_transformed,
																	data_type_output= DataType(6),
																	field='TERRITORY')
	
	categoricalToContinuous_input_dataDictionary_transformed=data_transformations.transform_cast_type(data_dictionary=categoricalToContinuous_input_dataDictionary_transformed,
																	data_type_output= DataType(6),
																	field='Instate')
	
	categoricalToContinuous_output_dataDictionary=categoricalToContinuous_input_dataDictionary_transformed
	categoricalToContinuous_output_dataDictionary.to_csv('./knime_dataDictionaries/stringToNumber_output_dataDictionary.csv')
	
	if contract_invariants.check_inv_missing_value_missing_value(data_dictionary_in=categoricalToContinuous_input_dataDictionary,
											data_dictionary_out=categoricalToContinuous_output_dataDictionary,
											belong_op_out=Belong(1), field_in='TERRITORY', field_out='TERRITORY'):
		print('INVARIANT INV_condition_TERRITORY VALIDATED')
	else:
		print('INVARIANT INV_condition_TERRITORY NOT VALIDATED')
	
	
	if contract_invariants.check_inv_missing_value_missing_value(data_dictionary_in=categoricalToContinuous_input_dataDictionary,
											data_dictionary_out=categoricalToContinuous_output_dataDictionary,
											belong_op_out=Belong(1), field_in='Instate', field_out='Instate'):
		print('INVARIANT INV_condition_Instate VALIDATED')
	else:
		print('INVARIANT INV_condition_Instate NOT VALIDATED')
	
	
#-----------------New DataProcessing-----------------
	imputeByNumericOp_input_dataDictionary=pd.read_csv('./knime_dataDictionaries/stringToNumber_output_dataDictionary.csv', sep=',')

	if contract_pre_post.check_outliers(belong_op=Belong(0), data_dictionary=imputeByNumericOp_input_dataDictionary, field='avg_income', 
									quant_abs=None, quant_rel=None, quant_op=None):
		print('PRECONDITION imputeOutlierByClosest(avg_income)_PRE_valueRange VALIDATED')
	else:
		print('PRECONDITION imputeOutlierByClosest(avg_income)_PRE_valueRange NOT VALIDATED')
	
	if contract_pre_post.check_outliers(belong_op=Belong(0), data_dictionary=imputeByNumericOp_input_dataDictionary, field='distance', 
									quant_abs=None, quant_rel=None, quant_op=None):
		print('PRECONDITION imputeOutlierByClosest(distance)_PRE_valueRange VALIDATED')
	else:
		print('PRECONDITION imputeOutlierByClosest(distance)_PRE_valueRange NOT VALIDATED')
	
	if contract_pre_post.check_outliers(belong_op=Belong(0), data_dictionary=imputeByNumericOp_input_dataDictionary, field='Instate', 
									quant_abs=None, quant_rel=None, quant_op=None):
		print('PRECONDITION imputeOutlierByClosest(Instate)_PRE_valueRange VALIDATED')
	else:
		print('PRECONDITION imputeOutlierByClosest(Instate)_PRE_valueRange NOT VALIDATED')
	
	imputeByNumericOp_input_dataDictionary_transformed=imputeByNumericOp_input_dataDictionary.copy()
	missing_values_list=[]
	
	imputeByNumericOp_input_dataDictionary_transformed=data_transformations.transform_special_value_num_op(data_dictionary=imputeByNumericOp_input_dataDictionary_transformed,
																  special_type_input=SpecialType(2), num_op_output=Operation(3),
																  missing_values=missing_values_list,		
																  axis_param=0, field_in = 'avg_income', field_out = 'avg_income')
	
	missing_values_list=[]
	
	imputeByNumericOp_input_dataDictionary_transformed=data_transformations.transform_special_value_num_op(data_dictionary=imputeByNumericOp_input_dataDictionary_transformed,
																  special_type_input=SpecialType(2), num_op_output=Operation(3),
																  missing_values=missing_values_list,		
																  axis_param=0, field_in = 'distance', field_out = 'distance')
	
	missing_values_list=[]
	
	imputeByNumericOp_input_dataDictionary_transformed=data_transformations.transform_special_value_num_op(data_dictionary=imputeByNumericOp_input_dataDictionary_transformed,
																  special_type_input=SpecialType(2), num_op_output=Operation(3),
																  missing_values=missing_values_list,		
																  axis_param=0, field_in = 'Instate', field_out = 'invalid')
	
	imputeByNumericOp_output_dataDictionary=imputeByNumericOp_input_dataDictionary_transformed
	imputeByNumericOp_output_dataDictionary.to_csv('./knime_dataDictionaries/numericOutliers_output_dataDictionary.csv')
	
	if contract_pre_post.check_outliers(belong_op=Belong(1), data_dictionary=imputeByNumericOp_output_dataDictionary, field='avg_income', 
									quant_abs=None, quant_rel=None, quant_op=None):
		print('POSTCONDITION imputeOutlierByClosest(avg_income)_POST_valueRange VALIDATED')
	else:
		print('POSTCONDITION imputeOutlierByClosest(avg_income)_POST_valueRange NOT VALIDATED')
	
	if contract_pre_post.check_outliers(belong_op=Belong(1), data_dictionary=imputeByNumericOp_output_dataDictionary, field='distance', 
									quant_abs=None, quant_rel=None, quant_op=None):
		print('POSTCONDITION imputeOutlierByClosest(distance)_POST_valueRange VALIDATED')
	else:
		print('POSTCONDITION imputeOutlierByClosest(distance)_POST_valueRange NOT VALIDATED')
	
	if contract_pre_post.check_outliers(belong_op=Belong(1), data_dictionary=imputeByNumericOp_output_dataDictionary, field='Instate', 
									quant_abs=None, quant_rel=None, quant_op=None):
		print('POSTCONDITION imputeOutlierByClosest(Instate)_POST_valueRange´ VALIDATED')
	else:
		print('POSTCONDITION imputeOutlierByClosest(Instate)_POST_valueRange´ NOT VALIDATED')
	
	if contract_invariants.check_inv_special_value_num_op(data_dictionary_in=imputeByNumericOp_input_dataDictionary,
											data_dictionary_out=imputeByNumericOp_output_dataDictionary,
											belong_op_in=Belong(0),
											belong_op_out=Belong(0),
											special_type_input=SpecialType(2),
											num_op_output=Operation(3),
											missing_values=None, axis_param=0, field_in='avg_income', field_out='avg_income'):
		print('INVARIANT imputeOutlierByClosest(avg_income)_INV_condition VALIDATED')
	else:
		print('INVARIANT imputeOutlierByClosest(avg_income)_INV_condition NOT VALIDATED')
	
	
	if contract_invariants.check_inv_special_value_num_op(data_dictionary_in=imputeByNumericOp_input_dataDictionary,
											data_dictionary_out=imputeByNumericOp_output_dataDictionary,
											belong_op_in=Belong(0),
											belong_op_out=Belong(0),
											special_type_input=SpecialType(2),
											num_op_output=Operation(3),
											missing_values=None, axis_param=0, field_in='distance', field_out='distance'):
		print('INVARIANT imputeOutlierByClosest(distance)_INV_condition VALIDATED')
	else:
		print('INVARIANT imputeOutlierByClosest(distance)_INV_condition NOT VALIDATED')
	
	
	if contract_invariants.check_inv_special_value_num_op(data_dictionary_in=imputeByNumericOp_input_dataDictionary,
											data_dictionary_out=imputeByNumericOp_output_dataDictionary,
											belong_op_in=Belong(0),
											belong_op_out=Belong(0),
											special_type_input=SpecialType(2),
											num_op_output=Operation(3),
											missing_values=None, axis_param=0, field_in='Instate', field_out='Instate'):
		print('INVARIANT imputeOutlierByClosest(Instate)_INV_condition VALIDATED')
	else:
		print('INVARIANT imputeOutlierByClosest(Instate)_INV_condition NOT VALIDATED')
	
	
#-----------------New DataProcessing-----------------
	binner_input_dataDictionary=pd.read_csv('./knime_dataDictionaries/numericOutliers_output_dataDictionary.csv', sep=',')

	if contract_pre_post.check_interval_range_float(left_margin=-1000.0, right_margin=1.0, data_dictionary=binner_input_dataDictionary,
	                                	closure_type=Closure(0), belong_op=Belong(0), field='TOTAL_CONTACTS'):
		print('PRECONDITION binner(TOTAL_CONTACTS)_PRE_valueRange VALIDATED')
	else:
		print('PRECONDITION binner(TOTAL_CONTACTS)_PRE_valueRange NOT VALIDATED')
	
	if contract_pre_post.check_interval_range_float(left_margin=-1000.0, right_margin=1.0, data_dictionary=binner_input_dataDictionary,
	                                	closure_type=Closure(0), belong_op=Belong(0), field='SELF_INIT_CNTCTS'):
		print('PRECONDITION binner(SELF_INIT_CNTCTS)_PRE_valueRange VALIDATED')
	else:
		print('PRECONDITION binner(SELF_INIT_CNTCTS)_PRE_valueRange NOT VALIDATED')
	
	if contract_pre_post.check_interval_range_float(left_margin=-1000.0, right_margin=1.0, data_dictionary=binner_input_dataDictionary,
	                                	closure_type=Closure(0), belong_op=Belong(0), field='SOLICITED_CNTCTS'):
		print('PRECONDITION binner(SOLICITED_CNTCTS)_PRE_valueRange VALIDATED')
	else:
		print('PRECONDITION binner(SOLICITED_CNTCTS)_PRE_valueRange NOT VALIDATED')
	
	binner_input_dataDictionary_transformed=binner_input_dataDictionary.copy()
	binner_input_dataDictionary_transformed=data_transformations.transform_derived_field(data_dictionary=binner_input_dataDictionary_transformed,
																  data_type_output = DataType(0),
																  field_in = 'TOTAL_CONTACTS', field_out = 'TOTAL_CONTACTS_binned')
	
	binner_output_dataDictionary=binner_input_dataDictionary_transformed
	binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
	binner_input_dataDictionary_transformed=data_transformations.transform_derived_field(data_dictionary=binner_input_dataDictionary_transformed,
																  data_type_output = DataType(0),
																  field_in = 'SELF_INIT_CNTCTS', field_out = 'SELF_INIT_CNTCTS_binned')
	
	binner_output_dataDictionary=binner_input_dataDictionary_transformed
	binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
	binner_input_dataDictionary_transformed=data_transformations.transform_derived_field(data_dictionary=binner_input_dataDictionary_transformed,
																  data_type_output = DataType(0),
																  field_in = 'SOLICITED_CNTCTS', field_out = 'SOLICITED_CNTCTS_binned')
	
	binner_output_dataDictionary=binner_input_dataDictionary_transformed
	binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
	binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																  left_margin=-1000.0, right_margin=1.0,
																  closure_type=Closure(0),
																  fix_value_output='low',
							                                      data_type_output = DataType(0),
																  field_in = 'TOTAL_CONTACTS',
																  field_out = 'TOTAL_CONTACTS_binned')
	
	binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																  left_margin=-1000.0, right_margin=1.0,
																  closure_type=Closure(0),
																  fix_value_output='low',
							                                      data_type_output = DataType(0),
																  field_in = 'SELF_INIT_CNTCTS',
																  field_out = 'SELF_INIT_CNTCTS_binned')
	
	binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																  left_margin=-1000.0, right_margin=1.0,
																  closure_type=Closure(0),
																  fix_value_output='low',
							                                      data_type_output = DataType(0),
																  field_in = 'SOLICITED_CNTCTS',
																  field_out = 'SOLICITED_CNTCTS_binned')
	
	binner_output_dataDictionary=binner_input_dataDictionary_transformed
	binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
	binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																  left_margin=1.0, right_margin=4.0,
																  closure_type=Closure(2),
																  fix_value_output='Moderate',
							                                      data_type_output = DataType(0),
																  field_in = 'TOTAL_CONTACTS',
																  field_out = 'TOTAL_CONTACTS_binned')
	
	binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																  left_margin=1.0, right_margin=4.0,
																  closure_type=Closure(2),
																  fix_value_output='Moderate',
							                                      data_type_output = DataType(0),
																  field_in = 'SELF_INIT_CNTCTS',
																  field_out = 'SELF_INIT_CNTCTS_binned')
	
	binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																  left_margin=1.0, right_margin=4.0,
																  closure_type=Closure(2),
																  fix_value_output='Moderate',
							                                      data_type_output = DataType(0),
																  field_in = 'SOLICITED_CNTCTS',
																  field_out = 'SOLICITED_CNTCTS_binned')
	
	binner_output_dataDictionary=binner_input_dataDictionary_transformed
	binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
	binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																  left_margin=4.0, right_margin=1000.0,
																  closure_type=Closure(2),
																  fix_value_output='High',
							                                      data_type_output = DataType(0),
																  field_in = 'TOTAL_CONTACTS',
																  field_out = 'TOTAL_CONTACTS_binned')
	
	binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																  left_margin=4.0, right_margin=1000.0,
																  closure_type=Closure(2),
																  fix_value_output='High',
							                                      data_type_output = DataType(0),
																  field_in = 'SELF_INIT_CNTCTS',
																  field_out = 'SELF_INIT_CNTCTS_binned')
	
	binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																  left_margin=4.0, right_margin=1000.0,
																  closure_type=Closure(2),
																  fix_value_output='High',
							                                      data_type_output = DataType(0),
																  field_in = 'SOLICITED_CNTCTS',
																  field_out = 'SOLICITED_CNTCTS_binned')
	
	binner_output_dataDictionary=binner_input_dataDictionary_transformed
	binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
	
	if contract_pre_post.check_interval_range_float(left_margin=-1000.0, right_margin=1.0, data_dictionary=binner_output_dataDictionary,
	                                	closure_type=Closure(0), belong_op=Belong(1), field='TOTAL_CONTACTS_binned'):
		print('POSTCONDITION binner(TOTAL_CONTACTS)_POST_valueRange VALIDATED')
	else:
		print('POSTCONDITION binner(TOTAL_CONTACTS)_POST_valueRange NOT VALIDATED')
	
	if contract_pre_post.check_interval_range_float(left_margin=-1000.0, right_margin=1.0, data_dictionary=binner_output_dataDictionary,
	                                	closure_type=Closure(0), belong_op=Belong(1), field='SELF_INIT_CNTCTS_binned'):
		print('POSTCONDITION binner(SELF_INIT_CNTCTS)_POST_valueRange VALIDATED')
	else:
		print('POSTCONDITION binner(SELF_INIT_CNTCTS)_POST_valueRange NOT VALIDATED')
	
	if contract_pre_post.check_interval_range_float(left_margin=-1000.0, right_margin=1.0, data_dictionary=binner_input_dataDictionary,
	                                	closure_type=Closure(0), belong_op=Belong(1), field='SOLICITED_CNTCTS'):
		print('POSTCONDITION binner(SOLICITED_CNTCTS)_POST_valueRange VALIDATED')
	else:
		print('POSTCONDITION binner(SOLICITED_CNTCTS)_POST_valueRange NOT VALIDATED')
	
	if contract_invariants.check_inv_interval_fix_value(data_dictionary_in=binner_input_dataDictionary,
											data_dictionary_out=binner_output_dataDictionary,
											left_margin=-1000.0, right_margin=1.0,
											closure_type=Closure(0),
											fix_value_output='Low',
											belong_op_in=Belong(0), belong_op_out=Belong(0),
											data_type_output=DataType(0),
											field_in='TOTAL_CONTACTS', field_out='TOTAL_CONTACTS_binned'):
		print('INVARIANT binner(TOTAL_CONTACTS)_INV_condition VALIDATED')
	else:
		print('INVARIANT binner(TOTAL_CONTACTS)_INV_condition NOT VALIDATED')
	if contract_invariants.check_inv_interval_fix_value(data_dictionary_in=binner_input_dataDictionary,
											data_dictionary_out=binner_output_dataDictionary,
											left_margin=1.0, right_margin=4.0,
											closure_type=Closure(2),
											fix_value_output='Moderate',
											belong_op_in=Belong(0), belong_op_out=Belong(0),
											data_type_output=DataType(0),
											field_in='TOTAL_CONTACTS', field_out='TOTAL_CONTACTS_binned'):
		print('INVARIANT binner(TOTAL_CONTACTS)_INV_condition VALIDATED')
	else:
		print('INVARIANT binner(TOTAL_CONTACTS)_INV_condition NOT VALIDATED')
	if contract_invariants.check_inv_interval_fix_value(data_dictionary_in=binner_input_dataDictionary,
											data_dictionary_out=binner_output_dataDictionary,
											left_margin=4.0, right_margin=1000.0,
											closure_type=Closure(2),
											fix_value_output='High',
											belong_op_in=Belong(0), belong_op_out=Belong(0),
											data_type_output=DataType(0),
											field_in='TOTAL_CONTACTS', field_out='TOTAL_CONTACTS_binned'):
		print('INVARIANT binner(TOTAL_CONTACTS)_INV_condition VALIDATED')
	else:
		print('INVARIANT binner(TOTAL_CONTACTS)_INV_condition NOT VALIDATED')
	
	
	if contract_invariants.check_inv_interval_fix_value(data_dictionary_in=binner_input_dataDictionary,
											data_dictionary_out=binner_output_dataDictionary,
											left_margin=-1000.0, right_margin=1.0,
											closure_type=Closure(0),
											fix_value_output='Low',
											belong_op_in=Belong(0), belong_op_out=Belong(0),
											data_type_output=DataType(0),
											field_in='SELF_INIT_CNTCTS', field_out='SELF_INIT_CNTCTS_binned'):
		print('INVARIANT INV_binner_condition_SELF_INIT_CNTCTS VALIDATED')
	else:
		print('INVARIANT INV_binner_condition_SELF_INIT_CNTCTS NOT VALIDATED')
	if contract_invariants.check_inv_interval_fix_value(data_dictionary_in=binner_input_dataDictionary,
											data_dictionary_out=binner_output_dataDictionary,
											left_margin=1.0, right_margin=4.0,
											closure_type=Closure(2),
											fix_value_output='Moderate',
											belong_op_in=Belong(0), belong_op_out=Belong(0),
											data_type_output=DataType(0),
											field_in='SELF_INIT_CNTCTS', field_out='SELF_INIT_CNTCTS_binned'):
		print('INVARIANT INV_binner_condition_SELF_INIT_CNTCTS VALIDATED')
	else:
		print('INVARIANT INV_binner_condition_SELF_INIT_CNTCTS NOT VALIDATED')
	if contract_invariants.check_inv_interval_fix_value(data_dictionary_in=binner_input_dataDictionary,
											data_dictionary_out=binner_output_dataDictionary,
											left_margin=4.0, right_margin=1000.0,
											closure_type=Closure(2),
											fix_value_output='High',
											belong_op_in=Belong(0), belong_op_out=Belong(0),
											data_type_output=DataType(0),
											field_in='SELF_INIT_CNTCTS', field_out='SELF_INIT_CNTCTS_binned'):
		print('INVARIANT INV_binner_condition_SELF_INIT_CNTCTS VALIDATED')
	else:
		print('INVARIANT INV_binner_condition_SELF_INIT_CNTCTS NOT VALIDATED')
	
	
	if contract_invariants.check_inv_interval_fix_value(data_dictionary_in=binner_input_dataDictionary,
											data_dictionary_out=binner_output_dataDictionary,
											left_margin=-1000.0, right_margin=1.0,
											closure_type=Closure(0),
											fix_value_output='Low',
											belong_op_in=Belong(0), belong_op_out=Belong(0),
											data_type_output=DataType(0),
											field_in='SOLICITED_CNTCTS', field_out='SOLICITED_CNTCTS_binned'):
		print('INVARIANT binner(SOLICITED_CNTCTS)_INV_condition VALIDATED')
	else:
		print('INVARIANT binner(SOLICITED_CNTCTS)_INV_condition NOT VALIDATED')
	if contract_invariants.check_inv_interval_fix_value(data_dictionary_in=binner_input_dataDictionary,
											data_dictionary_out=binner_output_dataDictionary,
											left_margin=1.0, right_margin=4.0,
											closure_type=Closure(2),
											fix_value_output='Moderate',
											belong_op_in=Belong(0), belong_op_out=Belong(0),
											data_type_output=DataType(0),
											field_in='SOLICITED_CNTCTS', field_out='SOLICITED_CNTCTS_binned'):
		print('INVARIANT binner(SOLICITED_CNTCTS)_INV_condition VALIDATED')
	else:
		print('INVARIANT binner(SOLICITED_CNTCTS)_INV_condition NOT VALIDATED')
	if contract_invariants.check_inv_interval_fix_value(data_dictionary_in=binner_input_dataDictionary,
											data_dictionary_out=binner_output_dataDictionary,
											left_margin=4.0, right_margin=1000.0,
											closure_type=Closure(2),
											fix_value_output='High',
											belong_op_in=Belong(0), belong_op_out=Belong(0),
											data_type_output=DataType(0),
											field_in='SOLICITED_CNTCTS', field_out='SOLICITED_CNTCTS_binned'):
		print('INVARIANT binner(SOLICITED_CNTCTS)_INV_condition VALIDATED')
	else:
		print('INVARIANT binner(SOLICITED_CNTCTS)_INV_condition NOT VALIDATED')
	
	
#-----------------New DataProcessing-----------------
	binner_input_dataDictionary=pd.read_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv', sep=',')

	if contract_pre_post.check_interval_range_float(left_margin=0.0, right_margin=1000.0, data_dictionary=binner_input_dataDictionary,
	                                	closure_type=Closure(3), belong_op=Belong(0), field='TERRITORY'):
		print('PRECONDITION binner(TERRITORY)_PRE_valueRange VALIDATED')
	else:
		print('PRECONDITION binner(TERRITORY)_PRE_valueRange NOT VALIDATED')
	
	binner_input_dataDictionary_transformed=binner_input_dataDictionary.copy()
	binner_input_dataDictionary_transformed=data_transformations.transform_derived_field(data_dictionary=binner_input_dataDictionary_transformed,
																  data_type_output = DataType(0),
																  field_in = 'TERRITORY', field_out = 'TERRITORY_binned')
	
	binner_output_dataDictionary=binner_input_dataDictionary_transformed
	binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
	binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																  left_margin=-1000.0, right_margin=1.0,
																  closure_type=Closure(0),
																  fix_value_output='Unknown',
							                                      data_type_output = DataType(0),
																  field_in = 'TERRITORY',
																  field_out = 'TERRITORY_binned')
	
	binner_output_dataDictionary=binner_input_dataDictionary_transformed
	binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
	binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																  left_margin=1.0, right_margin=3.0,
																  closure_type=Closure(2),
																  fix_value_output='Zone 1',
							                                      data_type_output = DataType(0),
																  field_in = 'TERRITORY',
																  field_out = 'TERRITORY_binned')
	
	binner_output_dataDictionary=binner_input_dataDictionary_transformed
	binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
	binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																  left_margin=3.0, right_margin=5.0,
																  closure_type=Closure(2),
																  fix_value_output='Zone 2',
							                                      data_type_output = DataType(0),
																  field_in = 'TERRITORY',
																  field_out = 'TERRITORY_binned')
	
	binner_output_dataDictionary=binner_input_dataDictionary_transformed
	binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
	binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																  left_margin=5.0, right_margin=7.0,
																  closure_type=Closure(2),
																  fix_value_output='Zone 3',
							                                      data_type_output = DataType(0),
																  field_in = 'TERRITORY',
																  field_out = 'TERRITORY_binned')
	
	binner_output_dataDictionary=binner_input_dataDictionary_transformed
	binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
	binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																  left_margin=7.0, right_margin=1000.0,
																  closure_type=Closure(3),
																  fix_value_output='Zone 4',
							                                      data_type_output = DataType(0),
																  field_in = 'TERRITORY',
																  field_out = 'TERRITORY_binned')
	
	binner_output_dataDictionary=binner_input_dataDictionary_transformed
	binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
	
	if contract_pre_post.check_interval_range_float(left_margin=0.0, right_margin=1000.0, data_dictionary=binner_output_dataDictionary,
	                                	closure_type=Closure(0), belong_op=Belong(1), field='TERRITORY_binned'):
		print('POSTCONDITION binner(TERRITORY)_POST_valueRange VALIDATED')
	else:
		print('POSTCONDITION binner(TERRITORY)_POST_valueRange NOT VALIDATED')
	
	if contract_invariants.check_inv_interval_fix_value(data_dictionary_in=binner_input_dataDictionary,
											data_dictionary_out=binner_output_dataDictionary,
											left_margin=-1000.0, right_margin=1.0,
											closure_type=Closure(0),
											fix_value_output='Unknown',
											belong_op_in=Belong(0), belong_op_out=Belong(0),
											data_type_output=DataType(0),
											field_in='TERRITORY', field_out='TERRITORY_binned'):
		print('INVARIANT binner(TERRITORY)_INV_condition VALIDATED')
	else:
		print('INVARIANT binner(TERRITORY)_INV_condition NOT VALIDATED')
	if contract_invariants.check_inv_interval_fix_value(data_dictionary_in=binner_input_dataDictionary,
											data_dictionary_out=binner_output_dataDictionary,
											left_margin=1.0, right_margin=3.0,
											closure_type=Closure(2),
											fix_value_output='Zone 1',
											belong_op_in=Belong(0), belong_op_out=Belong(0),
											data_type_output=DataType(0),
											field_in='TERRITORY', field_out='TERRITORY_binned'):
		print('INVARIANT binner(TERRITORY)_INV_condition VALIDATED')
	else:
		print('INVARIANT binner(TERRITORY)_INV_condition NOT VALIDATED')
	if contract_invariants.check_inv_interval_fix_value(data_dictionary_in=binner_input_dataDictionary,
											data_dictionary_out=binner_output_dataDictionary,
											left_margin=3.0, right_margin=5.0,
											closure_type=Closure(2),
											fix_value_output='Zone 2',
											belong_op_in=Belong(0), belong_op_out=Belong(0),
											data_type_output=DataType(0),
											field_in='TERRITORY', field_out='TERRITORY_binned'):
		print('INVARIANT binner(TERRITORY)_INV_condition VALIDATED')
	else:
		print('INVARIANT binner(TERRITORY)_INV_condition NOT VALIDATED')
	if contract_invariants.check_inv_interval_fix_value(data_dictionary_in=binner_input_dataDictionary,
											data_dictionary_out=binner_output_dataDictionary,
											left_margin=5.0, right_margin=7.0,
											closure_type=Closure(2),
											fix_value_output='Zone 3',
											belong_op_in=Belong(0), belong_op_out=Belong(0),
											data_type_output=DataType(0),
											field_in='TERRITORY', field_out='TERRITORY_binned'):
		print('INVARIANT binner(TERRITORY)_INV_condition VALIDATED')
	else:
		print('INVARIANT binner(TERRITORY)_INV_condition NOT VALIDATED')
	if contract_invariants.check_inv_interval_fix_value(data_dictionary_in=binner_input_dataDictionary,
											data_dictionary_out=binner_output_dataDictionary,
											left_margin=7.0, right_margin=1000.0,
											closure_type=Closure(2),
											fix_value_output='Zone 4',
											belong_op_in=Belong(0), belong_op_out=Belong(0),
											data_type_output=DataType(0),
											field_in='TERRITORY', field_out='TERRITORY_binned'):
		print('INVARIANT binner(TERRITORY)_INV_condition VALIDATED')
	else:
		print('INVARIANT binner(TERRITORY)_INV_condition NOT VALIDATED')
	
	
#-----------------New DataProcessing-----------------
	binner_input_dataDictionary=pd.read_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv', sep=',')

	if contract_pre_post.check_interval_range_float(left_margin=-1000.0, right_margin=2000.0, data_dictionary=binner_input_dataDictionary,
	                                	closure_type=Closure(3), belong_op=Belong(0), field='satscore'):
		print('PRECONDITION binner(satscore)_PRE_valueRange VALIDATED')
	else:
		print('PRECONDITION binner(satscore)_PRE_valueRange NOT VALIDATED')
	
	binner_input_dataDictionary_transformed=binner_input_dataDictionary.copy()
	binner_input_dataDictionary_transformed=data_transformations.transform_derived_field(data_dictionary=binner_input_dataDictionary_transformed,
																  data_type_output = DataType(0),
																  field_in = 'satscore', field_out = 'satscore_binned')
	
	binner_output_dataDictionary=binner_input_dataDictionary_transformed
	binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
	binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																  left_margin=-1000.0, right_margin=1040.0,
																  closure_type=Closure(1),
																  fix_value_output='54 Percentile and Under',
							                                      data_type_output = DataType(0),
																  field_in = 'satscore',
																  field_out = 'satscore_binned')
	
	binner_output_dataDictionary=binner_input_dataDictionary_transformed
	binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
	binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																  left_margin=1040.0, right_margin=1160.0,
																  closure_type=Closure(1),
																  fix_value_output='55-75 Percentile',
							                                      data_type_output = DataType(0),
																  field_in = 'satscore',
																  field_out = 'satscore_binned')
	
	binner_output_dataDictionary=binner_input_dataDictionary_transformed
	binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
	binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																  left_margin=1160.0, right_margin=1340.0,
																  closure_type=Closure(2),
																  fix_value_output='76-93 Percentile',
							                                      data_type_output = DataType(0),
																  field_in = 'satscore',
																  field_out = 'satscore_binned')
	
	binner_output_dataDictionary=binner_input_dataDictionary_transformed
	binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
	binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																  left_margin=1340.0, right_margin=2000.0,
																  closure_type=Closure(2),
																  fix_value_output='94+ percentile',
							                                      data_type_output = DataType(0),
																  field_in = 'satscore',
																  field_out = 'satscore_binned')
	
	binner_output_dataDictionary=binner_input_dataDictionary_transformed
	binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
	
	if contract_pre_post.check_interval_range_float(left_margin=-1000.0, right_margin=2000.0, data_dictionary=binner_output_dataDictionary,
	                                	closure_type=Closure(0), belong_op=Belong(1), field='satscore_binned'):
		print('POSTCONDITION binner(satscore)_POST_valueRange VALIDATED')
	else:
		print('POSTCONDITION binner(satscore)_POST_valueRange NOT VALIDATED')
	
	if contract_invariants.check_inv_interval_fix_value(data_dictionary_in=binner_input_dataDictionary,
											data_dictionary_out=binner_output_dataDictionary,
											left_margin=-1000.0, right_margin=1040.0,
											closure_type=Closure(1),
											fix_value_output='54 Percentile and Under',
											belong_op_in=Belong(0), belong_op_out=Belong(0),
											data_type_output=DataType(0),
											field_in='satscore', field_out='satscore_binned'):
		print('INVARIANT binner(satscore)_INV_condition VALIDATED')
	else:
		print('INVARIANT binner(satscore)_INV_condition NOT VALIDATED')
	if contract_invariants.check_inv_interval_fix_value(data_dictionary_in=binner_input_dataDictionary,
											data_dictionary_out=binner_output_dataDictionary,
											left_margin=1040.0, right_margin=1160.0,
											closure_type=Closure(1),
											fix_value_output='55-75 Percentile',
											belong_op_in=Belong(0), belong_op_out=Belong(0),
											data_type_output=DataType(0),
											field_in='satscore', field_out='satscore_binned'):
		print('INVARIANT binner(satscore)_INV_condition VALIDATED')
	else:
		print('INVARIANT binner(satscore)_INV_condition NOT VALIDATED')
	if contract_invariants.check_inv_interval_fix_value(data_dictionary_in=binner_input_dataDictionary,
											data_dictionary_out=binner_output_dataDictionary,
											left_margin=1160.0, right_margin=1340.0,
											closure_type=Closure(2),
											fix_value_output='76-93 Percentile',
											belong_op_in=Belong(0), belong_op_out=Belong(0),
											data_type_output=DataType(0),
											field_in='satscore', field_out='satscore_binned'):
		print('INVARIANT binner(satscore)_INV_condition VALIDATED')
	else:
		print('INVARIANT binner(satscore)_INV_condition NOT VALIDATED')
	if contract_invariants.check_inv_interval_fix_value(data_dictionary_in=binner_input_dataDictionary,
											data_dictionary_out=binner_output_dataDictionary,
											left_margin=1340.0, right_margin=2000.0,
											closure_type=Closure(2),
											fix_value_output='94+ percentile',
											belong_op_in=Belong(0), belong_op_out=Belong(0),
											data_type_output=DataType(0),
											field_in='satscore', field_out='satscore_binned'):
		print('INVARIANT binner(satscore)_INV_condition VALIDATED')
	else:
		print('INVARIANT binner(satscore)_INV_condition NOT VALIDATED')
	
	
#-----------------New DataProcessing-----------------
	binner_input_dataDictionary=pd.read_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv', sep=',')

	if contract_pre_post.check_interval_range_float(left_margin=9.0, right_margin=100000.0, data_dictionary=binner_input_dataDictionary,
	                                	closure_type=Closure(3), belong_op=Belong(0), field='avg_income'):
		print('PRECONDITION binner(avg_income)_PRE_valueRange VALIDATED')
	else:
		print('PRECONDITION binner(avg_income)_PRE_valueRange NOT VALIDATED')
	
	binner_input_dataDictionary_transformed=binner_input_dataDictionary.copy()
	binner_input_dataDictionary_transformed=data_transformations.transform_derived_field(data_dictionary=binner_input_dataDictionary_transformed,
																  data_type_output = DataType(0),
																  field_in = 'avg_income', field_out = 'avg_income_binned')
	
	binner_output_dataDictionary=binner_input_dataDictionary_transformed
	binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
	binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																  left_margin=-1000.0, right_margin=1040.0,
																  closure_type=Closure(1),
																  fix_value_output='Low',
							                                      data_type_output = DataType(0),
																  field_in = 'avg_income',
																  field_out = 'avg_income_binned')
	
	binner_output_dataDictionary=binner_input_dataDictionary_transformed
	binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
	binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																  left_margin=1040.0, right_margin=1160.0,
																  closure_type=Closure(1),
																  fix_value_output='Moderate',
							                                      data_type_output = DataType(0),
																  field_in = 'avg_income',
																  field_out = 'avg_income_binned')
	
	binner_output_dataDictionary=binner_input_dataDictionary_transformed
	binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
	binner_input_dataDictionary_transformed=data_transformations.transform_interval_fix_value(data_dictionary=binner_input_dataDictionary_transformed,
																  left_margin=1160.0, right_margin=1340.0,
																  closure_type=Closure(2),
																  fix_value_output='High',
							                                      data_type_output = DataType(0),
																  field_in = 'avg_income',
																  field_out = 'avg_income_binned')
	
	binner_output_dataDictionary=binner_input_dataDictionary_transformed
	binner_output_dataDictionary.to_csv('./knime_dataDictionaries/numericBinner_output_dataDictionary.csv')
	
	if contract_pre_post.check_interval_range_float(left_margin=9.0, right_margin=100000.0, data_dictionary=binner_output_dataDictionary,
	                                	closure_type=Closure(0), belong_op=Belong(1), field='avg_income_binned'):
		print('POSTCONDITION binner(avg_income)_POST_valueRange VALIDATED')
	else:
		print('POSTCONDITION binner(avg_income)_POST_valueRange NOT VALIDATED')
	
	if contract_invariants.check_inv_interval_fix_value(data_dictionary_in=binner_input_dataDictionary,
											data_dictionary_out=binner_output_dataDictionary,
											left_margin=9.0, right_margin=42830.0,
											closure_type=Closure(0),
											fix_value_output='Low',
											belong_op_in=Belong(0), belong_op_out=Belong(0),
											data_type_output=DataType(0),
											field_in='avg_income', field_out='avg_income_binned'):
		print('INVARIANT binner(avg_income)_INV_condition VALIDATED')
	else:
		print('INVARIANT binner(avg_income)_INV_condition NOT VALIDATED')
	if contract_invariants.check_inv_interval_fix_value(data_dictionary_in=binner_input_dataDictionary,
											data_dictionary_out=binner_output_dataDictionary,
											left_margin=42830.0, right_margin=5559.0,
											closure_type=Closure(2),
											fix_value_output='Moderate',
											belong_op_in=Belong(0), belong_op_out=Belong(0),
											data_type_output=DataType(0),
											field_in='avg_income', field_out='avg_income_binned'):
		print('INVARIANT binner(avg_income)_INV_condition VALIDATED')
	else:
		print('INVARIANT binner(avg_income)_INV_condition NOT VALIDATED')
	if contract_invariants.check_inv_interval_fix_value(data_dictionary_in=binner_input_dataDictionary,
											data_dictionary_out=binner_output_dataDictionary,
											left_margin=5559.0, right_margin=100000.0,
											closure_type=Closure(2),
											fix_value_output='High',
											belong_op_in=Belong(0), belong_op_out=Belong(0),
											data_type_output=DataType(0),
											field_in='avg_income', field_out='avg_income_binned'):
		print('INVARIANT binner(avg_income)_INV_condition VALIDATED')
	else:
		print('INVARIANT binner(avg_income)_INV_condition NOT VALIDATED')
	
	














set_logger("dataProcessing")
generateDataProcessing()
