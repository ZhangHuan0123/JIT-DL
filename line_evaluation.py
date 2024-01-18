import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def eval_result(result_df):
    ifa_lst, e20r_lst = [], []
    top_1_acc_lst, top_5_acc_lst, top_10_acc_lst = [], [], []
    top_5_prec_lst, top_10_prec_lst = [], []
    commits = result_df['commit'].unique()
    for commit_id in commits:
        cur_result = result_df[result_df['commit'] == commit_id]

        ifa, e20r, top_1_acc, top_5_acc, top_5_prec = get_line_level_metrics(cur_result['score'].tolist(), cur_result['label'].tolist())

        ifa_lst.append(ifa)
        e20r_lst.append(e20r)
        top_1_acc_lst.append(top_1_acc)
        top_5_acc_lst.append(top_5_acc)
        top_5_prec_lst.append(top_5_prec)

    print(round(np.mean(ifa_lst), 4),
          round(np.mean(e20r_lst), 4),
          round(np.mean(top_1_acc_lst), 4),
          round(np.mean(top_5_acc_lst), 4),
          round(np.mean(top_5_prec_lst), 4),
          )


def get_line_level_metrics(line_score, label):
    scaler = MinMaxScaler()
    line_score = scaler.fit_transform(np.array(line_score).reshape(-1, 1))

    line_df = pd.DataFrame()
    line_df['scr'] = [float(val) for val in list(line_score)]
    line_df['label'] = label
    line_df = line_df.sort_values(by='scr', ascending=False)
    line_df['row'] = np.arange(1, len(line_df) + 1)

    real_buggy_lines = line_df[line_df['label'] == 1]

    top_1_acc, top_5_acc = 0, 0
    top_5_prec = 0
    if len(real_buggy_lines) < 1:
        ifa = len(line_df)
        e20r = math.ceil(0.2 * len(line_df))
    else:
        ifa = line_df[line_df['label'] == 1].iloc[0]['row'] - 1
        label_list = list(line_df['label'])

        top_1_acc = 1 if np.sum(label_list[:1]) > 0 else 0
        top_5_acc = 1 if np.sum(label_list[:5]) > 0 else 0

        top_5_prec = np.sum(label_list[:5]) / len(label_list[:5])

        buggy_20_percent = real_buggy_lines.head(math.ceil(0.2 * len(real_buggy_lines)))
        buggy_20_percent_row_num = buggy_20_percent.iloc[-1]['row']
        e20r = int(buggy_20_percent_row_num) / float(len(line_df))

    return ifa, e20r, top_1_acc, top_5_acc, top_5_prec


def main(path):
    result_df = pd.read_csv(path, sep='\t')
    eval_result(result_df)


if __name__ == "__main__":
    path = "ours_af.txt"
    main(path)