import datetime
import pandas as pd
import numpy as np
import logging

import os

os.chdir('D:\\src\\Stock-Research')


def check_factor_data(path, data_type=None):
    """检查因子数据是否符合格式要求
    path : 文件路径
    data_type : 目前多个岗位提交的因子集格式不一样, 所以需要分开检查 , 默认None:研究员岗位 , 'AI':ai岗位 , 'T0':T0岗位
    返回 : 如果检查到异常返回ValuError , 否则返回None
    """
    DUP_LIMIT = 0.5  # 缺失/重复值的比例上限
    STOCK_LIMIT = 1700  # 涵盖的股票个数 , 当前上市的股票有5000多
    DATA_RANGE_BEGIN = 730  # 数据最小的日期距离当前日期的天数
    DATA_RANEG_END = 90  # 数据最大的日期距离当前日期的天数

    # 读取文件
    try:
        datas = pd.read_csv(path, index_col=0)
    except Exception as e:
        raise ValueError(
            "文件格式校验失败，无法正常读取文件，请检查数据格式，文件必须是csv格式或csv.gz压缩文件(换行符'\n' , 分隔符为',')，错误信息: {}".format(
                e))
    logging.info(f"read data end  , lenght {len(datas)}")
    # 检查日期列
    error_timeline = datas[~datas.index.astype(str).str.match(r'^\d{4}([-/]?)(\d{1,2}\1\d{1,2}|\d{2}\d{2})$')].index[:5]
    if not error_timeline.empty:
        raise ValueError(
            "文件格式校验失败，日期列异常：数据集的第一列必须是类如'2023-01-02'的年月日格式，异常日期格式:{}".format(
                error_timeline.astype(str).tolist()))

    if data_type:
        if data_type.lower() == 'ai':
            logging.info("check ai data")
            times = datas.pop(datas.columns[0])
            error_tline = times[~times.isin(['10:00:00', '13:30:00'])][:5]
            if not error_tline.empty:
                raise ValueError(
                    "文件格式校验失败，时间列异常：数据集的第二列必须是'10:00:00','13:30:00'中的一个，异常时间列:{}".format(
                        error_tline.astype(str).tolist()))


        elif data_type == 't0':
            logging.info("check t0 data")
            times = datas.pop(datas.columns[0])
            error_tline = times[~times.astype(str).str.match(r'^\d{2}:\d{2}:\d{2}$')]
            if not error_tline.empty:
                raise ValueError(
                    "文件格式校验失败，时间列异常：数据集的第二列必须形如'%H:%M:%S'的time格式，异常时间列:{}".format(
                        error_tline.astype(str).tolist()))
    else:
        check_date = datetime.date.today()
        begin_date = check_date - datetime.timedelta(days=DATA_RANGE_BEGIN)
        end_date = check_date - datetime.timedelta(days=DATA_RANEG_END)

        # 检查数据开始范围
        dates = pd.to_datetime(datas.index).date
        bdate = dates.min()
        if bdate > begin_date:
            raise ValueError(
                "文件格式校验失败，数据集过少：预期数据开始日期至少为{} ,实际为 {}".format(begin_date, bdate))

        edate = dates.max()
        if edate < end_date:
            raise ValueError("文件格式校验失败，数据集过少：预期数据结束日期至少为{} ,实际为 {}".format(end_date, edate))

            # 检查标的列数(股票个数)
    if datas.shape[1] < STOCK_LIMIT:
        raise ValueError(f"文件格式校验失败，数据集异常：覆盖股票数量过少")

    # 检查标的行
    error_stocks = datas.columns[~datas.columns.astype(str).str.match(r'^((68|60)\d{4}\.XSHG|(30|00)\d{4}\.XSHE)$')][
                   :10]
    if not error_stocks.empty:
        raise ValueError(
            "文件格式校验失败，标的代码异常：数据集的第一行必须是类如'000001.XSHE'/'600000.XSHG' 的标的代码(上交所后缀.XSHG，深交所后缀.XSHE，不支持其他交易所)，异常代码格式:{}".format(
                error_stocks.astype(str).tolist()))

    # 数据类型检查
    try:
        datas = datas.astype(np.float64)
    except ValueError as e:
        raise ValueError("文件格式校验失败，数据集异常：除第一行及第一列外，其他行列均应为数字或空：{}".format(e))
    except Exception as e:
        raise ValueError("文件格式校验失败，数据集异常：数据无法转换为数字类型：{}".format(e))

    # 有效值(去重,缺失)比例检查,检查的是每天的平均值
    real_count = datas.fillna(999888777666).nunique(axis=1)
    dup_ratio = 1 - (real_count / datas.shape[1]).mean()
    if dup_ratio > DUP_LIMIT:
        raise ValueError(
            "文件格式校验失败，预测值缺失/重复检查异常：预测值缺失/重复占比不应该超过{}，实际为: {:.2f}".format(DUP_LIMIT, dup_ratio))


if __name__ == '__main__':
    check_factor_data("model/simpleMLP_compare/pred/feat200_epoch30_corr0.5910_submit_result.csv", data_type='AI')
    print("检查通过")
