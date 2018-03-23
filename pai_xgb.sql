drop table if exists qrl_xgboost_pred;
DROP OFFLINEMODEL IF EXISTS qrl_xgboost_1;

-- train
PAI
-name xgboost
-project algo_public
-Deta="0.02"
-Dobjective="reg:linear"
-DitemDelimiter=","
-Dseed="0"
-Dnum_round="2000"
-DlabelColName="sale_201701"
-DinputTableName="pai_temp_112507_1268526_1"
-DenableSparse="false"
-Dmax_depth="8"
-Dsubsample="0.6"
-Dcolsample_bytree="0.7"
-DmodelName="qrl_xgboost_1"
-Dgamma="0"
-Dlambda="30" 
-DfeatureColNames="nums,province_id,city_id,type_id,sale_201612,sale_201611,sale_201610,avg1,avg2,avg3,avg4,avg5,avg6,avg7,avg8,avg9,avg10,avg11,cityfeature01,wm1,wm2,wm3,wm4,wm5,wm6,wm7,wm8,wm9,wm10,wm11,std1,std2,std3,brand_id,department_id,if_mpv_id,if_luxurious_id,level_id"
-Dbase_score="0.5"
-Dmin_child_weight="100"
-DkvDelimiter=":";


-- predict
PAI
-name prediction
-project algo_public
-DdetailColName="prediction_detail"
-DappendColNames="province_id,city_id,class_id,sale_201701"
-DmodelName="qrl_xgboost_1"
-DitemDelimiter=","
-DresultColName="prediction_result"
-Dlifecycle="28"
-DoutputTableName="qrl_xgboost_pred"
-DscoreColName="prediction_score"
-DkvDelimiter=":"
-DfeatureColNames="nums,province_id,city_id,type_id,sale_201612,sale_201611,sale_201610,avg1,avg2,avg3,avg4,avg5,avg6,avg7,avg8,avg9,avg10,avg11,cityfeature01,wm1,wm2,wm3,wm4,wm5,wm6,wm7,wm8,wm9,wm10,wm11,std1,std2,std3,brand_id,department_id,if_mpv_id,if_luxurious_id,level_id"
-DinputTableName="pai_temp_112507_1268526_2"
-DenableSparse="false";

--评估
drop table if exists qrl_index_output_table;
drop table if exists qrl_residual_output_table;
pai
-name regression_evaluation 
-project algo_public 
-DinputTableName=qrl_xgboost_pred 
-DyColName=sale_201701 
-DpredictionColName=prediction_score 
-DindexOutputTableName=qrl_index_output_table
-DresidualOutputTableName=qrl_residual_output_table;
  

  
  