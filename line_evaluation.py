import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def eval_result(result_df):
    IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall = [], [], []
    top_10_acc, top_5_acc, top_1_acc = [], [], []
    commits = result_df['commit'].unique()
    for commit_id in commits:
        cur_result = result_df[result_df['commit'] == commit_id]

        cur_IFA, \
        cur_top_20_percent_LOC_recall, \
        cur_effort_at_20_percent_LOC_recall, \
        cur_top_10_acc, \
        cur_top_5_acc, \
        cur_top_1_acc = get_line_level_metrics(cur_result['score'].tolist(), cur_result['label'].tolist())

        IFA.append(cur_IFA)
        top_20_percent_LOC_recall.append(cur_top_20_percent_LOC_recall)
        effort_at_20_percent_LOC_recall.append(cur_effort_at_20_percent_LOC_recall)
        top_10_acc.append(cur_top_10_acc)
        top_5_acc.append(cur_top_5_acc)
        top_1_acc.append(cur_top_1_acc)

    print(round(np.mean(top_1_acc), 4),
          round(np.mean(top_5_acc), 4),
          # round(np.mean(top_10_acc), 4),
          round(np.mean(top_20_percent_LOC_recall), 4),
          round(np.mean(effort_at_20_percent_LOC_recall), 4),
          round(np.mean(IFA), 4))


def get_line_level_metrics(line_score, label):
    scaler = MinMaxScaler()
    line_score = scaler.fit_transform(np.array(line_score).reshape(-1, 1))

    line_df = pd.DataFrame()
    line_df['scr'] = [float(val) for val in list(line_score)]
    line_df['label'] = label
    line_df = line_df.sort_values(by='scr', ascending=False)
    line_df['row'] = np.arange(1, len(line_df) + 1)

    real_buggy_lines = line_df[line_df['label'] == 1]

    top_10_acc = 0
    top_5_acc = 0
    top_1_acc = 0
    if len(real_buggy_lines) < 1:
        IFA = len(line_df)
        top_20_percent_LOC_recall = 0
        effort_at_20_percent_LOC_recall = math.ceil(0.2 * len(line_df))
    else:
        IFA = line_df[line_df['label'] == 1].iloc[0]['row'] - 1
        label_list = list(line_df['label'])

        all_rows = len(label_list)

        # find top-10 & top-5 accuracy
        if all_rows < 10:
            top_10_acc = np.sum(label_list[:all_rows]) / len(label_list[:all_rows])
        else:
            top_10_acc = np.sum(label_list[:10]) / len(label_list[:10])
        if all_rows < 5:
            top_5_acc = np.sum(label_list[:all_rows]) / len(label_list[:all_rows])
        else:
            top_5_acc = np.sum(label_list[:5]) / len(label_list[:5])
        top_1_acc = label_list[0]

        # find recall
        LOC_20_percent = line_df.head(int(0.2 * len(line_df)))
        buggy_line_num = LOC_20_percent[LOC_20_percent['label'] == 1]
        top_20_percent_LOC_recall = float(len(buggy_line_num)) / float(len(real_buggy_lines))

        # find effort @20% LOC recall
        buggy_20_percent = real_buggy_lines.head(math.ceil(0.2 * len(real_buggy_lines)))
        buggy_20_percent_row_num = buggy_20_percent.iloc[-1]['row']
        assert int(buggy_20_percent_row_num) <= float(len(line_df))
        effort_at_20_percent_LOC_recall = int(buggy_20_percent_row_num) / float(len(line_df))

    return IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc, top_1_acc


def main(path):
    result_df = pd.read_csv(path, sep='\t')
    eval_result(result_df)

print('Top-1-ACC Top-5-ACC R20%E E@20%R IFA')
main(path="../DeepDL/data/deepdl_line_result.txt")
# for i in range(1, 31):
#     for j in range(1, 11):
#         print(i,j*400)
#         main(path="data/y_0_{}_{}.txt".format(str(i), str(j*400)))
