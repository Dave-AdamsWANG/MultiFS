
import torch
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from models.emb_MLPs import *
from itertools import permutations
from dataset import Movielens1MDataset
import pickle
import time

def get_dataset(path):
    return Movielens1MDataset(path)


def get_model(name, args, decision):
    if name == 'MLP':
        return MLP(args,decision)



class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model.state_dict(), self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (field, target) in enumerate(tk0):
        field, target = field.to(device), target.to(device)
        y = model(field)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for field, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            field, target = field.to(device), target.to(device)
            y = model(field)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts)


def main(dataset_path,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         save_dir,
         conf,
         args):
    device = torch.device(device)
    dataset = get_dataset(dataset_path)
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length), generator=torch.Generator().manual_seed(42))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)
    decisions = make_decisions(drop=1) + make_decisions(drop = 2) + make_decisions(drop = 3) + \
            make_decisions(drop=4) + make_decisions(drop =5) + [[1]*8]
    test_auc = {}
    start = time.time()
    for decision in decisions:
        print(decision)
        model = get_model(model_name, args, decision).to(device)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        early_stopper = EarlyStopper(num_trials=2, save_path=f'{save_dir}/{model_name}.pt')
        for epoch_i in range(epoch):
            train(model, optimizer, train_data_loader, criterion, device)
            auc = test(model, valid_data_loader, device)
            print('epoch:', epoch_i, 'validation: auc:', auc)
            if not early_stopper.is_continuable(model, auc):
                print(f'validation: best auc: {early_stopper.best_accuracy}')
                break
        auc = test(model, test_data_loader, device)
        test_auc[str(decision)] = auc
    cost = (time.time()-start)/3600
    with open('test_auc_tmp_seed42%.2f.pkl'%cost,'wb') as f:
        pickle.dump(test_auc,f)
    
    


def make_decisions(total = 8, drop = 1):
    decisions = []
    decision = [1] * (total-drop) + [0] * drop
    decisions = decisions + list(set(permutations(decision)))
    return decisions 

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_outputs', type=int, help='output dimension', default=1)
    parser.add_argument('--inner_representation_size', type=int, help='output size of mixing linear layers', default=16)
    parser.add_argument('--batchnorm', help='Use batch norm', action='store_true', default=False)
    parser.add_argument("--drpt", action="store", default=0.5, dest="drpt", type=float, help="dropout")
    parser.add_argument('--dataset_path', default='/data/wangyejing/Field/ml-1m/train.txt')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--model_name', default='MLP')
    parser.add_argument('--conf', type=int, default=[[3,4,0]])
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:3')
    parser.add_argument('--save_dir', default='/data/wangyejing/Field/chkpt')
    parser.add_argument('--embed_dim', type=float, default=16)
    parser.add_argument('--mlp_dims', type=float, default=(16, 8))
    parser.add_argument('--dropout', type=int, default=0.2)
    args = parser.parse_args()
    args.field_dims = [3706,301,81,6040,21,7,2,3402]
    main(args.dataset_path,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.device,
         args.save_dir,
         args.conf,
         args)
