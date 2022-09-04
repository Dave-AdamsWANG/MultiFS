import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, log_loss
import time

from torch.utils.data import DataLoader
from dataset import *
from pruner import *
from torchfm.layer import *


def get_dataset(name, path):
    if name == 'movielens1M':
        return Movielens1MDataset(path)
    elif name == 'criteo':
        return CriteoDataset(path)
    elif name == 'avazu':
        return AvazuDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)



class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy,emb=None):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save({'state_dict': model.state_dict()}, self.save_path)  # torch.save(model, self.save_path)
            if emb:
                torch.save({'state_dict': emb.state_dict()}, 'chkpt/emb.pt')
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def train(model, optimizer, data_loader, criterion, device, log_interval=100,emb=None):
    model.train()
    total_loss = 0
    tk0 = tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        fields, target = fields.to(device), target.to(device)
        if emb:
            embed_x = emb(fields)
            y = model(fields,embed_x)
        else:
            y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0

def test(model, data_loader, device,emb=None):
    model.eval()
    targets, predicts, infer_time  = list(), list(), list()
    with torch.no_grad():
        for fields, target in tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            start = time.time()
            if emb:
                embed_x = emb(fields)
                y = model(fields,embed_x)
            else:
                y = model(fields)
            infer_cost = time.time() - start
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
            infer_time.append(infer_cost)
    return roc_auc_score(targets, predicts), log_loss(targets, predicts), sum(infer_time)

def main(dataset_name,
         dataset_path,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         save_dir,args):
    device = torch.device(device)
    dataset = get_dataset(dataset_name, dataset_path)
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length),generator=torch.Generator().manual_seed(42))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size)
    emb = FeaturesEmbedding(args.field_dims,args.embed_dim)
    model = get_model(model_name, args.field_dims, args.embed_dim).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopper(num_trials=2, save_path=f'{save_dir}/{model_name}.pt')
    #********************************  pre-train  ********************************#
    for epoch_i in range(args.pre_epoch):
        train(model, optimizer, train_data_loader, criterion, device,emb=emb)
        auc, logloss, infer_time = test(model, valid_data_loader,device,emb=emb)
        print('epoch:', epoch_i, 'validation: auc:', auc,'logloss:',logloss,'infer_time:', infer_time)
        if not early_stopper.is_continuable(model, auc,emb=emb):
            print(f'validation: best auc: {early_stopper.best_accuracy}')
            break
    model.load_state_dict(torch.load(f'{save_dir}/{model_name}.pt')['state_dict'])
    emb.load_state_dict(torch.load(f'{save_dir}/emb.pt')['state_dict'])
    auc, logloss, infer_time = test(model, test_data_loader, device, emb=emb)
    print('Pretrain Result/n','test: auc:', auc,'logloss:',logloss,'infer_time:', infer_time)
    #********************************  Pruning  ********************************#
    pruner = Pruner(emb,model,criterion,train_data_loader)
    masks = pruner(compression_factor=args.compress)[0]
    del pruner
    pruned_model = PrunedModel(emb,model_name, args.field_dims, masks)
    del emb 
    del model
    #********************************  Retraining  ********************************#
    print(masks)
    early_stopper = EarlyStopper(num_trials=2, save_path=f'{save_dir}/{model_name}_pruned.pt')
    for epoch_i in range(epoch):
        train(pruned_model, optimizer, train_data_loader, criterion, device)
        auc, logloss, infer_time = test(pruned_model, valid_data_loader,device)
        print('epoch:', epoch_i, 'validation: auc:', auc,'logloss:',logloss,'infer_time:', infer_time)
        if not early_stopper.is_continuable(pruned_model, auc):
            print(f'validation: best auc: {early_stopper.best_accuracy}')
            break
    pruned_model.load_state_dict(torch.load(f'{save_dir}/{model_name}_pruned.pt')['state_dict'])
    auc, logloss, infer_time = test(pruned_model, test_data_loader, device)
    print('Pretrain Result/n','test: auc:', auc,'logloss:',logloss,'infer_time:', infer_time)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='movielens1M')
    parser.add_argument('--dataset_path', help='criteo/train.txt, avazu/train, or ml-1m/ratings.dat')
    parser.add_argument('--model_name', default='dfm')
    parser.add_argument('--pre_epoch', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--embed_dim',type=int,default=8)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--compress',default=0.5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='chkpt')

    args = parser.parse_args()
    if args.dataset_name == 'criteo': args.dataset_path = 'criteo/train.txt'
    if args.dataset_name == 'avazu': args.dataset_path = '/Users/wangyejing/Desktop/FS/dataset/Avazu/train'
    if args.dataset_name == 'movielens1M': args.dataset_path = '/Users/wangyejing/Desktop/FS/dataset/ml-1m/train.txt'
    if args.dataset_name == 'movielens1M':
        args.field_dims = [3706,301,81,6040,21,7,2,3402]
    elif args.dataset_name == 'avazu':
        args.field_dims = [241, 8, 8, 3697, 4614, 25, 5481, 329, 
            31, 381763, 1611748, 6793, 6, 5, 2509, 9, 10, 432, 5, 68, 169, 61]
    main(args.dataset_name,
         args.dataset_path,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.device,
         args.save_dir, args)
