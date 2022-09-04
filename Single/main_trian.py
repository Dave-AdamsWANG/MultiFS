import torch
import tqdm, gc, time
from sklearn.metrics import roc_auc_score, log_loss
from torch.utils.data import DataLoader
from models.emb_MLPs import *

from dataset import AvazuDataset, Movielens1MDataset, CriteoDataset


def get_dataset(name, path):
    if name == 'movielens1M' or name == 'movielens1M_inter':
        return Movielens1MDataset(path)
    elif name == 'avazu':
        return AvazuDataset(path)
    elif name == 'criteo':
        return CriteoDataset(path)


def get_model(name,args):
    if name == 'MLP':
        return select_MLP(args)


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
            if model.stage == 1:
                model.desicion = decide(model)
            torch.save({'state_dict': model.state_dict()}, self.save_path)  # torch.save(model, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False
        # Load model: model = describe_model(); checkpoint = torch.load('checkpoint.pth.tar'); model.load_state_dict(checkpoint['state_dict'])


def train(model, optimizer, optimizer_model, optimizer_darts, train_data_loader, valid_data_loader, criterion, device, log_interval, AutoML, darts_frequency):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(train_data_loader, smoothing=0, mininterval=1.0)
    valid_data_loader_iter = iter(valid_data_loader)
    #min_space_ratio, best_decision = 1.0, []
    # val_fields, val_target = [], []
    for i, (fields, target) in enumerate(tk0):
        # if model.stage == 1: val_fields.append(fields); val_target.append(target)
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        # for layer_name, param in model.named_parameters(): print('0', layer_name, param[0])

        # supervised training, jointly update main RS network and Darts weights
        if not AutoML:
            optimizer.step()

        # pretrain/test stage, only update main RS network
        if AutoML and model.stage in [0, 2]:
            optimizer_model.step()

        # search stage, alternatively update main RS network and Darts weights
        if AutoML and model.stage == 1:
            optimizer_model.step()
            if (i + 1) % darts_frequency == 0:
                # fields, target = torch.cat(val_fields, 0), torch.cat(val_target, 0); val_fields, val_target = [], []
                try:
                    fields, target = next(valid_data_loader_iter)
                except StopIteration:
                    del valid_data_loader_iter
                    gc.collect()
                    valid_data_loader_iter = iter(valid_data_loader)
                    fields, target = next(valid_data_loader_iter)
                fields, target = fields.to(device), target.to(device)
                y = model(fields)
                loss_val = criterion(y, target.float())

                model.zero_grad()
                loss_val.backward()
                optimizer_darts.step()

        # if ((i + 1) % (darts_frequency * 50) == 0 or i == len(tk0) - 1) and model.stage == 1:
        #     decision = decide(model)
        #     if ratio < min_space_ratio:
        #         min_space_ratio = ratio
        #         best_decision = decision

        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0
        # if i > 100: break



def test(model, data_loader, device):
    model.eval()
    targets, predicts, infer_time  = list(), list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            start = time.time()
            y = model(fields)
            infer_cost = time.time() - start
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
            infer_time.append(infer_cost)
    return roc_auc_score(targets, predicts), log_loss(targets, predicts), sum(infer_time)



def decide(model):
    dec_f = model.dec_f
    if dec_f == 0:
        p = torch.softmax(model.weights.deep_weights,dim=0)
        # _, index = p.sort(dim=0,descending=True)
        # index = index.detach().cpu().numpy()
        # decision = [0] * len(index)
        # for i in index[:11]:
        #     decision[i] =1
        base = p[-1]
        decision = []
        for i in p[:-1]:
            if i > base:
                decision.append(1)
            else:
                decision.append(0)
    if dec_f == 1:
    # SoftMAX selection with drop n
        _, index = torch.softmax(model.weights.deep_weights,dim=0)[:,-1].sort(dim=0,descending=True)
        index = index.detach().cpu().numpy()
        decision = [0] * len(index)
        num = 11#int(len(index)/2) # 8,10,11 is not ok for criteo 
        for i in index[:num]:
            decision[i] =1
    if dec_f == 2:
    # Hard Selection
        decision = torch.argmax(model.weights.deep_weights, dim=1).detach().cpu().numpy()
    return decision

def main(dataset_name,
         dataset_path,
         model_name,
         args,
         epoch,
         learning_rate,
         learning_rate_darts,
         batch_size,
         AutoML,
         darts_frequency,
         weight_decay,
         device,
         save_dir):
    device = torch.device(device)
    dataset = get_dataset(dataset_name, dataset_path)
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length), generator=torch.Generator().manual_seed(42))

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size*2, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)

    model = get_model(model_name, args).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_model = torch.optim.Adam(params=[param for name, param in model.named_parameters() if name != 'weights.deep_weights'], lr=learning_rate, weight_decay=weight_decay)
    optimizer_darts = torch.optim.Adam(params=[param for name, param in model.named_parameters() if name == 'weights.deep_weights'], lr=learning_rate_darts, weight_decay=weight_decay)

    # print('\n*********************************************Pretrain*********************************************\n')
    # model.stage = 0
    # early_stopper = EarlyStopper(num_trials=3, save_path=f'{save_dir}/{model_name}:{model.stage}.pt')
    # for epoch_i in range(epoch[0]):
    #     print('Pretrain epoch:', epoch_i)
    #     train(model, optimizer, optimizer_model, optimizer_darts, train_data_loader, valid_data_loader, criterion, device, 100, AutoML, darts_frequency)

    #     auc, logloss,infer_time = test(model, valid_data_loader, device)
    #     if not early_stopper.is_continuable(model, auc):
    #         print(f'validation: best auc: {early_stopper.best_accuracy}')
    #         break
    #     print('Pretrain epoch:', epoch_i, 'validation: auc:', auc, 'logloss:', logloss)

    # auc, logloss,infer_time = test(model, test_data_loader, device)
    # print(f'Pretrain test auc: {auc} logloss: {logloss}, infer time:{infer_time}\n')

    print('\n********************************************* Search *********************************************\n')
    model.stage = 1
    early_stopper = EarlyStopper(num_trials=3, save_path=f'{save_dir}/{model_name}:{model.stage}.pt')
    for epoch_i in range(epoch[1]):
        print('Search epoch:', epoch_i)
        train(model, optimizer, optimizer_model, optimizer_darts, train_data_loader, valid_data_loader, criterion, device, 100, AutoML, darts_frequency)
        auc, logloss,_ = test(model, valid_data_loader, device)
        print(model.weights.deep_weights)
        print(f'decision = {decide(model)}')
        if auc > early_stopper.best_accuracy:decision_best = decide(model)
        if not early_stopper.is_continuable(model, auc):
            print(f'validation: best auc: {early_stopper.best_accuracy}')
            print(f'decision = {decision_best}')
            break
        
        print('Search epoch:', epoch_i, 'validation: auc:', auc, 'logloss:', logloss)

        # if epoch_i > 0 and ratio < min_space_ratio:  # epoch 0 is not stable
        #     min_space_ratio, best_decision = ratio, decision
        # print(f'min_space_ratio = {min_space_ratio}, best_decision = {best_decision}')
        

    auc, logloss, _ = test(model, test_data_loader, device)
    print(f'Search test auc: {auc} logloss: {logloss}\n')
    
    #params = sum(np.array(decision) * args.field_dims)/sum(args.field_dims)
    #print(params)
    print('\n*********************************************  Test  *********************************************\n')
    time_retrain = time.time()
    model.stage = 2
    model = MLP(args,decision_best).to(device)
    optimizer_model = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopper(num_trials=3, save_path=f'{save_dir}/{model_name}_test.pt')
    for epoch_i in range(epoch[2]):
        print('Test epoch:', epoch_i)
        train(model, optimizer, optimizer_model, optimizer_darts, train_data_loader, valid_data_loader, criterion, device, 100, AutoML, darts_frequency)
        auc, logloss, infer_time = test(model, valid_data_loader, device)
        
        print('Test epoch:', epoch_i, 'validation: auc:', auc, 'logloss:', logloss,'infer time:', infer_time)

        if not early_stopper.is_continuable(model, auc):
            print(f'Test validation: best auc: {early_stopper.best_accuracy}')
            break

    auc, logloss,infer_time = test(model, test_data_loader, device)
    print(f'Test test auc: {auc} logloss: {logloss} infer time: {infer_time}\n ')
    print(model.decision)
    print(f'Retrain Time:{time.time()-time_retrain}')
    with open('Record/%s.txt'%dataset_name, 'a') as the_file:
        the_file.write('Dataset:%s\nRetrain Time:%.2f,Retrain Epoches: %d\n Decision:%s,\ntest auc:%.8f,logloss:%.8f, infer time:%.8f\n'
        %(dataset_name,(time.time()-time_retrain)/60, epoch_i+1, str(model.decision),auc,logloss,infer_time))



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='movielens1M',
                        help='criteo, avazu, movielens1M, movielens20M, movielens1Mfield_noBinary, movielens1Mfield_noGenres, movielens1Mfield_all')
    parser.add_argument('--dataset_path', default='ml-1m/train.txt',
                        help='criteo/train.txt, avazu/train, ml-1m/ratings.dat, ml-20m/rating.csv, ml-1m/train_field_no_binary.txt, ml-1m/train_field_no_genres.txt, ml-1m/train_field_all.txt')
    parser.add_argument('--model_name', default='MLP', help='lr, fm, hofm, ffm, fnn, wd, ipnn, opnn, dcn, fnfm, dfm, xdfm, afm, afi, afn, nfm, ncf')
    parser.add_argument('--mlp_dims', type=int, default=[16,8], help='original=16')
    parser.add_argument('--embed_dim', type=int, default=16, help='original=16')
    parser.add_argument('--softmax_type', type=int, default=0, help='0 softmax; 1 softmax+temperature; 2 gumbel-softmax')
    parser.add_argument('--epoch', type=int, default=[0,8,50], nargs='+', help='pretrain/search/test epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--learning_rate_darts', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--AutoML', type=bool, default=True)
    parser.add_argument('--darts_frequency', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--dropout',type=int, default=0.2)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', help='cuda:0')
    parser.add_argument('--save_dir', default='chkpt')
    parser.add_argument('--decide_f',default=1, help='Decide funtion to use. 1: Hand select, 2: auto select by argmax')
    parser.add_argument('--add_zero',default=False, help='Whether to add a useless feature')
    args = parser.parse_args()

    if args.dataset_name == 'criteo': dataset_path = 'criteo/train.txt'
    if args.dataset_name == 'avazu': dataset_path = 'avazu/train'
    if args.dataset_name == 'movielens1M': dataset_path = 'ml-1m/train.txt'
    if args.dataset_name == 'movielens1M_inter': dataset_path = 'ml-1m/train_inter_part_noid.txt'
    if args.dataset_name == 'movielens1M':
        args.field_dims = [3706,301,81,6040,21,7,2,3402]
    elif args.dataset_name == 'movielens1M_inter':
        args.field_dims = [3706, 301, 81, 6040, 21, 7, 2, 3402, 1524, 6100, 2076, 601, 1648, 561, 162,134, 42, 5349, 14, 4794, 4079]
        #[3706, 301, 81, 6040, 21, 7, 2, 3402, 59112, 22457, 7152, 1524, 6100, 2076, 601, 268050, 1648, 561, 162,130340,134, 42, 5349, 14, 4794, 4079]
        #[3706,301,81,6040,21,7,2,3402,3706,3706,1000209,59112,22457,7152,871096,1524,352172,6100,2076,601,268050,195382,1648,561,162,130340,6040,6040,6040,6040,134,42,5349,14,4794,4079]
    elif args.dataset_name == 'avazu':
        args.field_dims = [241, 8, 8, 3697, 4614, 25, 5481, 329, 
            31, 381763, 1611748, 6793, 6, 5, 2509, 9, 10, 432, 5, 68, 169, 61]
    elif args.dataset_name == 'criteo':
        args.field_dims = [    49,    101,    126,     45,    223,    118,     84,     76,
           95,      9,     30,     40,     75,   1458,    555, 193949,
       138801,    306,     19,  11970,    634,      4,  42646,   5178,
       192773,   3175,     27,  11422, 181075,     11,   4654,   2032,
            5, 189657,     18,     16,  59697,     86,  45571]
    if args.add_zero:
        args.field_dims.append(0)
    print(f'\ndataset_name = {args.dataset_name},\t',
          f'dataset_path = {dataset_path},\t',
          f'model_name = {args.model_name},\t',
          f'mlp_dim = {args.mlp_dims},\t',
          f'softmax_type = {args.softmax_type},\t',
          f'epoch = {args.epoch},\t',
          f'learning_rate = {args.learning_rate},\t',
          f'learning_rate_darts = {args.learning_rate_darts},\t',
          f'batch_size = {args.batch_size},\t',
          f'AutoML = {args.AutoML},\t',
          f'darts_frequency = {args.darts_frequency},\t',
          f'weight_decay = {args.weight_decay},\t',
          f'device = {args.device},\t',
          f'save_dir = {args.save_dir}\n')
    for i in range(10):
        time_start = time.time()
        main(args.dataset_name,
            dataset_path,
            args.model_name,
            args,
            args.epoch,
            args.learning_rate,
            args.learning_rate_darts,
            args.batch_size,
            args.AutoML,
            args.darts_frequency,
            args.weight_decay,
            args.device,
            args.save_dir)

        print(f'\ndataset_name = {args.dataset_name},\t',
            f'dataset_path = {dataset_path},\t',
            f'model_name = {args.model_name},\t',
            f'mlp_dim = {args.mlp_dims},\t',
            f'softmax_type = {args.softmax_type},\t',
            f'epoch = {args.epoch},\t',
            f'learning_rate = {args.learning_rate},\t',
            f'learning_rate_darts = {args.learning_rate_darts},\t',
            f'batch_size = {args.batch_size},\t',
            f'AutoML = {args.AutoML},\t',
            f'darts_frequency = {args.darts_frequency},\t',
            f'weight_decay = {args.weight_decay},\t',
            f'device = {args.device},\t',
            f'save_dir = {args.save_dir},\t',
            f'training time = {(time.time() - time_start) / 3600}\n')