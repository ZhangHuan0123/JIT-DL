import os
import argparse
import torch
import time
import torch.nn as nn
from tqdm import tqdm
import pickle
from ours import Model
from nt_xent import NT_Xent
from transformers import RobertaTokenizer, RobertaModel
from utils import load_data, train_batch, test_batch


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-load_model", type=str,
                        help='Loading our model for evaluation.')

    parser.add_argument("-train", action='store_true',
                        help="Whether to run training.")

    parser.add_argument("-test", action='store_true',
                        help="Whether to run testing.")

    parser.add_argument("-batch_size", type=int, default=8,
                        help="Batch size per GPU/CPU for training.")

    parser.add_argument("-lr", type=float, default=1e-5,
                        help="The initial learning rate for Adam.")

    parser.add_argument("-max_token", type=int, default=20,
                        help="The max length of a code line corresponding to a node in a graph")

    parser.add_argument('-epochs', type=int, default=3,
                        help="The number of epochs for training.")

    parser.add_argument('-in_dim', type=int, default=512,
                        help='The dimension of embedding vector for a sequence encoder')

    parser.add_argument('-hid_dim', type=int, default=1024,
                        help='The dimension of embedding vector for a graph encoder')

    parser.add_argument('-n_layers', type=int, default=2,
                        help="The number of the code graph layer")

    parser.add_argument('-eps', type=float, default=0.4,
                        help="The number of the code graph layer")

    parser.add_argument('-dropout', type=float, default=0.5,
                        help="The dropout rate")

    parser.add_argument('-device', type=int, default=1,
                        help='Device to use for iterate data')

    parser.add_argument('-seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('-save_dir', type=str, default="model",
                        help="The dropout rate")

    parser.add_argument('-name', type=str, default="ours",
                        help="The model name")

    params = parser.parse_args()
    return params


def train(train_data, bert, params):
    model = Model(params, bert).to(params.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    criterion_1 = nn.BCELoss(reduce=False)
    criterion_2 = NT_Xent(params.batch_size, 0.1, 1)
    model.train()
    for epoch in range(1, params.epochs + 1):
        total_loss, step = 0, 0
        batches = train_batch(train_data, batch_size=params.batch_size, seed=params.seed)
        for batch in tqdm(batches):
            step += 1
            commits, files, graphs, labels = batch
            input_ids = torch.tensor(graphs.input_ids).to(params.device)
            input_masks = torch.tensor(graphs.input_masks).to(params.device)
            g_0 = torch.tensor(graphs.g_0).to(params.device)
            g_1 = torch.tensor(graphs.g_1).to(params.device)
            g_2 = torch.tensor(graphs.g_2).to(params.device) if graphs.g_2 is not None else None
            labels = torch.FloatTensor(labels).to(params.device)
            target_ids = graphs.target_ids
            add_ids = graphs.add_ids

            predicts, g_feat1, g_feat2 = model.forward(input_ids,
                                                       input_masks,
                                                       g_0, g_1, g_2,
                                                       target_ids,
                                                       add_ids,
                                                       pertub=True)
            bce_loss = criterion_1(predicts, labels)
            bce_loss = bce_loss.mean()
            cl_loss = 0.05 * criterion_2(g_feat1, g_feat2)
            loss = bce_loss + cl_loss
            total_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
            if step % 400 == 0:
                print("Epoch:%i Step:%i Loss:%f" % (epoch, step, total_loss))
                total_loss = 0
                path = params.save_dir + params.name + "_" + str(epoch) + "_" + str(step) + ".pt"
                torch.save(model.state_dict(), path)


def test(test_data, bert, params):
    batches = test_batch(test_data, batch_size=params.batch_size)
    model = Model(params, bert).to(params.device)
    model.load_state_dict(torch.load(params.load_model))
    model.eval()
    with torch.no_grad():
        all_labels, all_add_ids, all_add_labels, all_add_nums = [], [], [], []
        all_preds = []
        for batch in batches:
            commits, files, graphs, labels = batch
            input_ids = torch.tensor(graphs.input_ids).to(params.device)
            input_masks = torch.tensor(graphs.input_masks).to(params.device)
            g_0 = torch.tensor(graphs.g_0).to(params.device)
            g_1 = torch.tensor(graphs.g_1).to(params.device)
            g_2 = torch.tensor(graphs.g_2).to(params.device) if graphs.g_2 is not None else None
            target_ids = graphs.target_ids
            add_ids = graphs.add_ids
            preds, attns = model.forward(input_ids,
                                            input_masks,
                                            g_0, g_1, g_2,
                                            target_ids,
                                            add_ids,
                                            pertub=False)

            preds = preds.cpu().detach().numpy().tolist()
            attns = attns.cpu().detach().numpy()
            all_preds.extend(preds)
            all_labels += labels
        pickle.dump([all, all_labels], open("min.pkl", "wb"))


if __name__ == "__main__":
    params = parse_args()
    bert_path = "../../codebert-base"
    tokenizer = RobertaTokenizer.from_pretrained(bert_path)
    bert = RobertaModel.from_pretrained(bert_path)

    params.device = torch.device("cuda:{}".format(params.device) if torch.cuda.is_available() else "cpu")
    if params.train:
        train_cache = 'data/train_cache.pkl'
        if os.path.exists(train_cache):
            train_data = pickle.load(open(train_cache, 'rb'))
        else:
            train_data = load_data(path="data/train.pkl", tokenizer=tokenizer, max_tokens=params.max_token)
            pickle.dump(train_data, open(train_cache, 'wb'))
        train(train_data, bert, params)
    elif params.test:
        start_time = time.time()
        test_cache = 'data/test_cache.pkl'
        if os.path.exists(test_cache):
            test_data = pickle.load(open(test_cache, 'rb'))
        else:
            test_data = load_data(path="data/test.pkl", tokenizer=tokenizer, max_tokens=params.max_token)
            pickle.dump(test_data, open(test_cache, 'wb'))
        test(test_data, bert, params)
        end_time = time.time()
        print(end_time - start_time)
    else:
        print('-----------------Something wrongs with your command--------------------')
        exit()
