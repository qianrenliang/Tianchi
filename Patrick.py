#!/usr/bin/env python
#coding:utf-8

# 区域订单预测
# 场景描述：利用滴滴的历史订单数据去预测未来一天24小时的订单数据
# 策略概述：利用了目标日期的前8周和前5天的每小时对应的数据以及前4小时的数据，总共利用了13 * 5=65个历史数据
# 还利用了前8周对应的均值，星期几的标注，isholiday，hourid ,type >>>total 70 features
# 单独建模，划分规则：每个城市，根据小区的某天的小时的均值订单进行划分，单独建模分析
# // A ，B 2个小区，A小区和B小区的每小时平均订单在[0,10]之间，属于一个等级，这2个小区的数据就划分到一个区间
# 共分为6个等级，24个小时，共计6*24=144个子模型
# 不足之处：天气特征未加入，缺失值的处理有待商榷

# 待改进：修改xgboost的loss function


import sys
import os,os.path
import numpy as np
import xgboost as xgb
import cPickle as pickle
import math
import lightgbm as lgb
from matplotlib import pyplot
from statistics import median,mean


def mae_obj(preds, dtrain):
    labels = dtrain.get_label()
    res = np.array(preds - labels)
    grad = (np.exp(2 * res) - 1) / (np.exp(2 * res) + 1)
    # hess = 4 * np.exp(2 * res) / np.power((np.exp(2 * res) + 1), 2)
    hess = np.repeat([1], len(res))
    return grad, hess



class ProcessData:
    '处理类'
    def __init__(self, weekspan, dayspan, hourspan, time_span, flag):
        self.weeks = weekspan
        self.days = dayspan
        self.hours = hourspan
        self.time_span = time_span
        self.flag = flag
        self.samples = {}
        self.levels = [0, 5, 10, 20, 100, 10000000]
        # [0,5] [5,10] [10,20] [20,100] [100,1000000]

    def get_hex_stat(self, areacube): # 获取小区等级
        tmp = []
        for day, dayvalue in areacube.items(): # 某个小区每一天的数据
            for hour, hourvalue in dayvalue.items(): # 某一天的每个小时的数据
                tmp.append(hourvalue['ordercnt'])
        stat = sum(tmp)/len(tmp) # 某个小区的平均订单数/时
        for idx in range(len(self.levels)-1):
            if self.levels[idx] <= stat < self.levels[idx+1]:
                return idx




    def get_hour_mean(self, areacube):
        tmp = {}
        for day, dayvalue in areacube.items():
            for hour, hourvaule in dayvalue.items():
                tmp.setdefault(hour, [])
                tmp[hour].append(hourvaule['ordercnt'])
        mean_value = {}
        for i in tmp.keys():
            mean_value[i] = sum(tmp[i]) / len(tmp[i])
        return mean_value


    def median_filter(self, series, windows):
        res = []
        for i in range(len(series)):
            if i < windows - 1:
                res.append(median(series[:i + 1]))
            else:
                res.append(median(series[i - windows + 1:i + 1]))
        return res


    def max_filter(self, series, windows):
        res = []
        for i in range(len(series)):
            if i < windows - 1:
                res.append(max(series[:i+1]))
            else:
                res.append(max(series[i - windows + 1: i+1]))
        return res

    def mean_filter(self, series, windows):
        res = []
        for i in range(len(series)):
            if i < windows -1:
                res.append(mean(series[:i+1]))
            else:
                res.append(mean(series[i - windows + 1: i+1]))
        return res


    def print_out(self, lastcity, lastareaid, areacube):
        #输出到文件
        print "processing... city=%s, hex_id=%s" % (lastcity, lastareaid)
        stat_level = self.get_hex_stat(areacube)

        mean_values = self.get_hour_mean(areacube)

        print lastcity, lastareaid, stat_level
        for day, dayvalue in areacube.items():
            #历史小时没数据可以用0弥补
            # --------------------------------------------------------------------------------
            historydays = set() # 集合
            weekdays = set()    # weekday存储前8周的dayid
            for daydeta in range(self.weeks, 0, -1):
                # daydeta = [8,7,6,5,4,3,2,1]
                targetday = day - daydeta*7
                weekdays.add(targetday)
                # 前8周的对应日子的数字
                historydays.add(targetday)
            for daydeta in range(self.days, 0, -1):
                # daydeta = [5,4,3,2,1]
                targetday = day - daydeta
                # 前5日
                historydays.add(targetday)
            historydays = sorted(historydays)   # 有前5日的dayid和前8周的dayid，共有13个dayid，且不重复

            median_days = set()
            for i in range(16, 0 ,-1):
                targetday = day - i * 7
                median_days.add(targetday)
            median_days = sorted(median_days)

            for hour, hourvalue  in dayvalue.items():
                timeseries = [] # 记录某一天的每个小时的前5天的对应的平均订单数
                weekday_hour_value = [] # 记录前8周同一个工作日同一个hourid的订单数，计算平均得到timeseries[0]
                median_serie = []
                # ----------------------------------------------
                for median_day in median_days:
                    if not median_day in areacube or not hour not in areacube[day]:
                        median_serie.append(0)
                    else:
                        median_serie.append(areacube[median_day][hour][flag])
                median_series = self.median_filter(median_serie,3)
                max_series = self.max_filter(median_serie,3)
                mean_series = self.mean_filter(median_serie, 3)
                # ------------------------------------------------------------
                for weekday in weekdays:
                    if not weekday in areacube or not hour in areacube[weekday]:
                        # ----------changed
                        # weekday_hour_value.append(0)
                        weekday_hour_value.append(mean_values[hour])
                    else:
                        weekday_hour_value.append(areacube[weekday][hour][flag])
                mean_value = sum(weekday_hour_value)/len(weekday_hour_value) # 某一天对应的前8周的平均订单数
                timeseries.append(mean_value)
                for historyday in historydays:
                    # self.hours = 5
                    for d in range(self.hours-1 , -1, -1): # historydays里的13个dayid对应时间的数据和之前4个小时的数据
                        # d = [4,3,2,1,0]
                        realhour = hour - d
                        realday = historyday
                        if realhour < 0:
                            realhour += 24
                            realday -= 1
                        if realday not in areacube or realhour not in areacube[realday]:
                            # timeseries.append(0)
                            timeseries.append(mean_values[hour])
                        else:
                            timeseries.append(areacube[realday][realhour][flag])
                        # 一个dayid 对应5个数据,1+13*5 = 66
                # 序列构建完毕
                # timeseries：{前8周均值，historydays里对应的前5个小时的数据(not mean):13*5}
                # sample : [真实值,timeseries,week,holiday,type,hour]

                sample = [hourvalue[flag]]
                sample += timeseries
                sample += median_series
                sample += max_series
                sample += mean_series
                sample += [hourvalue['week'], hourvalue['holiday'], hourvalue['type'], hour]
                # 1个sample 有1+66+4 = 71个数据，1个label，70个features

                # self.samples={}
                self.samples.setdefault(lastcity, {})
                self.samples[lastcity].setdefault(stat_level, {})
                self.samples[lastcity][stat_level].setdefault(hour, {})
                self.samples[lastcity][stat_level][hour].setdefault(day, [])
                self.samples[lastcity][stat_level][hour][day].append(sample)



    def process(self, infile):
        lastcity = ''  # 上一个城市id
        lastareaid = '' # 上一个地区id
        areacube = {} # 地区信息
        count = 0
        for line in infile:
            if count % 100000 == 0:
                # 每处理一万条数据，输出一次信息
                print count
            count += 1
            terms = line.strip().split('\t')
            cityid = terms[0]  # 城市id
            time_span = terms[3]  # 时间跨度
            if not cityid.isdigit() or self.time_span != time_span:
                continue
            areaid = terms[1] # hex_id
            dayid = terms[7] # dayid
            hourid = terms[4] # hourid [0-23]
            order_cnt = float(terms[5]) # 订单数
            finish_order_cnt = float(terms[6]) # 订单完成数
            isweek = int(terms[8])
            isholiday = int(terms[9])
            thetype = int(terms[10])
            # 新的小区数据
            if areaid != lastareaid:

                if not lastareaid: # 第一个小区
                    areacube = {}
                    lastcity = cityid
                    lastareaid = areaid
                    # --------------------------------------------------------------
                    if int(dayid) not in areacube:
                            areacube[int(dayid)] = {}
                    if order_cnt > 0:
                            finish_ratio = round(finish_order_cnt/order_cnt, 4)
                    else:
                        finish_ratio = 0.0
                    areacube[int(dayid)][int(hourid)] = {'ordercnt':order_cnt, 'finish':finish_order_cnt, 'finish_ratio':finish_ratio,
                                                         'week': isweek, 'holiday': isholiday, 'type': thetype}
                    continue
                    #----------------------------------------------------------------
                self.print_out(lastcity, lastareaid, areacube)  # 产出数据
                areacube = {}
                lastcity = cityid
                lastareaid = areaid

            if int(dayid) not in areacube:
                areacube[int(dayid)] = {}
            if order_cnt > 0:
                finish_ratio = round(finish_order_cnt/order_cnt, 4)
            else:
                finish_ratio = 0.0
            areacube[int(dayid)][int(hourid)] = {'ordercnt':order_cnt, 'finish':finish_order_cnt, 'finish_ratio':finish_ratio,
                                                 'week': isweek, 'holiday': isholiday, 'type': thetype}
        # last city
        self.print_out(lastcity, lastareaid, areacube)
        return 0




    def maybe_pickle(self, infile, force=False):
        """对象的序列化，方便程序下次启动时不需要再处理数据"""
        pickle_file = "%s.pickle" % (infile)
        self.pickle_file = pickle_file
        if os.path.exists(self.pickle_file) and not force:
            print('%s already present - Skipping pickling.' % pickle_file)
        else:
            print('Pickling %s.' % pickle_file)
            self.process(open(infile))
            try:

                with open(pickle_file, 'wb') as f:
                    pickle.dump(self.samples, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', pickle_file, ':', e)


    def cal_error(self, result, thre):
        """计算平均绝对百分比误差"""
        mapes = []
        for predict, label in result:
            if label>thre:
                mape = abs(predict - label)/label
                mapes.append(mape)
        return sum(mapes)/len(mapes)


    def trans_log(self, x):
        try:
            res = math.log(x, math.e)
        except:
            res = 0.0
        return round(res, 4)


    def trans_exp(self, x):
        try:
            res = math.exp(x)
        except:
            res = 0
        return round(res, 4)






    def train(self):

        # with open(self.pickle_file, 'rb') as f:
        #     print('Load data from %s' % self.pickle_file)
        #     self.samples = pickle.load(f)

        training_errors, testing_errors, baseline_errors = [], [], []
        # self.samples>>>lastcity , stat_level, hour, day
        for city, city_values in self.samples.items():
            for level, level_values in city_values.items():
                for hour, hour_values in level_values.items():
                    print "training city=%s, level=%s, hour=%s" % (city, level, hour)
                    training, testing = [], []
                    for day, day_values in hour_values.items():

                        if 100<=day<=230:
                            training += day_values
                        if 230<day<=237:
                            testing += day_values
                        # x rows 71 cols
                        # cols: ordercnt, timeseries(mean_value, 13 different dayid * 5 different hourid's value),
                        # week, isholiday, type, hourid[0-23]
                    print 'training.shape: %s, testing.shape: %s' % (len(training), len(testing))

                    # features:13 different dayid * 5 different hourid's values + 8 weeks ago mean value + weekid + isholiday + type?? + hourid
                    # log(label)


                    training_x, training_y = np.array(training)[:, 1:], np.array(training)[:, 0]
                    testing_x, testing_y = np.array(testing)[:, 1:], np.array(testing)[:, 0]

                    baseline_y = np.array(testing)[:, 1] # mean_value

                    training_y = map(self.trans_log, training_y)
                    testing_y = map(self.trans_log, testing_y)
                    training_y, testing_y = np.array(training_y), np.array(testing_y)


                    print 'training_x.shape: %s, training_y.shape: %s' % (training_x.shape, training_y.shape)
                    print 'testing_x.shape: %s, testing_y.shape: %s' % (testing_x.shape, testing_y.shape)


                    #                                xgb_process(X,Y,testX,testY,weight,depth,iterations,colsample_bytree)
                    reg, predict_x, predict_y = self.xgb_process(training_x, training_y, testing_x, testing_y, 0.5, 4, 500, 0.8)
                    #                                lgb_process(trainX,trainY,testX,testY,depth,iterations,num_leaf)
                    # reg, predict_x, predict_y = self.lgb_process(training_x, training_y, testing_x, testing_y, 4, 500, 10)
                    # predict_x >> train set predict res   predict_y >> test set predict res


                    predict_x = map(self.trans_exp, predict_x)
                    predict_y = map(self.trans_exp, predict_y)
                    training_y = map(self.trans_exp, training_y)
                    testing_y = map(self.trans_exp, testing_y)




                    for value in zip(predict_y, testing_y, baseline_y):
                        print "pre_value: %s\t%s\t%s" % (city, level, "\t".join([str(x) for x in value]))


                    training_error = self.cal_error(zip(predict_x, training_y), 0)
                    testing_error = self.cal_error(zip(predict_y, testing_y), 0)
                    baseline_error = self.cal_error(zip(baseline_y, testing_y), 0)

                    training_errors.append([training_error*len(training_y), len(training_y)])
                    testing_errors.append([testing_error*len(testing_y), len(testing_y)])
                    baseline_errors.append([baseline_error*len(testing_y), len(testing_y)])
                    print "city=%s, level=%s, hour=%s, training error: %.3f, testing error: %.3f, baseline error: %.3f" % (city, level, hour, training_error, testing_error, baseline_error)
                    # error is mae
            print "city=%s, total error, training error: %.3f, testing error: %.3f, baseline error: %.3f" % \
                (
                city,
                sum([x[0] for x in training_errors])/sum([x[1] for x in training_errors]),
                sum([x[0] for x in testing_errors])/sum([x[1] for x in testing_errors]),
                sum([x[0] for x in baseline_errors])/sum([x[1] for x in baseline_errors])
                )
            print "city=%s, level 1+ error, training error: %.3f, testing error: %.3f, baseline error: %.3f" % \
                (
                city,
                sum([x[0] for x in training_errors[24:]])/sum([x[1] for x in training_errors[24:]]),
                sum([x[0] for x in testing_errors[24:]])/sum([x[1] for x in testing_errors[24:]]),
                sum([x[0] for x in baseline_errors[24:]])/sum([x[1] for x in baseline_errors[24:]])
                )
            print "city=%s, level 2+ error, training error: %.3f, testing error: %.3f, baseline error: %.3f" % \
                (
                city,
                sum([x[0] for x in training_errors[48:]])/sum([x[1] for x in training_errors[48:]]),
                sum([x[0] for x in testing_errors[48:]])/sum([x[1] for x in testing_errors[48:]]),
                sum([x[0] for x in baseline_errors[48:]])/sum([x[1] for x in baseline_errors[48:]])
                )



    def xgb_process(self, X, Y, testX, testY, weight, depth, iterations, colsample_bytrxee):
        training_set = xgb.DMatrix(X, label=Y, missing=np.nan)
        testing_set = xgb.DMatrix(testX, label=testY, missing=np.nan)
        watch_list = [(training_set, 'train'), (testing_set, 'eval')]
        param = {
            'eta': 0.1,
            # 'gamma': 0.1,
            'max_depth': depth,
            # 'min_child_weight': 1,
            # 'max_delta_step': 0,
            'subsample': 0.8,
            'colsample_bytrxee': colsample_bytrxee,
            # 'colsample_bylevel': 1,
            # 'lambda': 1, >> l2
             'lambda': 0.1,
            # 'tree_method': 'auto',
            # 'sketch_eps': 0.03,
            # 'scale_pos_weight': 1 / weight,
            'objective': 'reg:linear',
            'eval_metric': 'mae',
            'seed': 31,
            'silent': 1
        }

        eval_dic = {}
        tree_booster = xgb.train(params=param, dtrain=training_set, num_boost_round=iterations, evals=watch_list,
                                 evals_result=eval_dic, early_stopping_rounds=50, obj=mae_obj)  # , learning_rates=decay)
        # print tree_booster.best_iteration, '\t', tree_booster.best_ntree_limit

        predict_y = tree_booster.predict(testing_set, output_margin=False, ntree_limit=tree_booster.best_iteration)
        predict_x = tree_booster.predict(training_set, output_margin=False, ntree_limit=tree_booster.best_iteration)

        # feature_importance = tree_booster.get_score(importance_type='gain')

        # epochs = len(eval_dic['eval']['mae'])
        # x_axis = range(0, epochs)
        # fig ,ax = pyplot.subplots()
        # ax.plot(x_axis, eval_dic['train']['mae'], label='Train')
        # ax.plot(x_axis, eval_dic['eval']['mae'], label='Test')
        # ax.legend()
        # pyplot.ylabel("Mae")
        # pyplot.title("Xgboost Mae")
        # pyplot.show()



        return (tree_booster, predict_x, predict_y)


    def lgb_process(self, trainX, trainY, testX, testY, depth, iterations, num_leaf):
        trainset = lgb.Dataset(data=trainX, label=trainY)
        testset = lgb.Dataset(data=testX, label=testY, reference=trainset)

        param = {
            'learning_rate': 0.1,
            'objective': 'regression_l2',
            'num_leaves': num_leaf,
            'max_depth': depth,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.8,
            'feature_fraction_seed': 2,
            'bagging_fraction': 0.8,
            'bagging_freq': 10,
            'lambda_l2': 0.1,
            # 'boosting': 'gbdt',
            'metric': 'mae'

        }

        eval_dic = {}
        bst = lgb.train(param, trainset, num_boost_round=iterations, valid_sets=[trainset,testset],
                        valid_names=['train', 'eval'], evals_result=eval_dic)


        predict_y = bst.predict(trainX)
        predict_x = bst.predict(testX)

        return (bst, predict_x, predict_y)


if __name__ == "__main__":
    weekspan = 8
    dayspan = 5
    hourspan = 5
    time_span = "3600"
    flag = "ordercnt"
    pd = ProcessData(weekspan, dayspan, hourspan, time_span, flag)

    # 1M rows
    infile = "C:\Users\qccc\Desktop\mini.txt"
    print "load data"
    pd.process(open(infile))
    # pd.maybe_pickle(infile)
    print "load data done"
    print "start training"
    pd.train()
    print "end training"


# result : >>>>>>
# 未取log之前:
# city=1, total error, training error: 0.447, testing error: 0.478, baseline error: 0.479
# city=1, >= 5 level 1+ error, training error: 0.368, testing error: 0.396, baseline error: 0.378
# city=1, >=10 level 2+ error, training error: 0.316, testing error: 0.329, baseline error: 0.314

# 取log后：
# city=1, total error, training error: 0.358, testing error: 0.399, baseline error: 0.485
# city=1, level 1+ error, training error: 0.302, testing error: 0.347, baseline error: 0.378
# city=1, level 2+ error, training error: 0.263, testing error: 0.296, baseline error: 0.314
