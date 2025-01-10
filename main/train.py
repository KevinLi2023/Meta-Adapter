import torch
from torch.optim import AdamW
from torch.nn import functional as F
from avalanche.evaluation.metrics.accuracy import Accuracy
from tqdm import tqdm
from timm.models import create_model
from timm.scheduler.cosine_lr import CosineLRScheduler
from argparse import ArgumentParser
from vtab import *
from utils import *
from MetaAdapter import set_MetaAdapter
from json_dataset import get_fgvc_data

def train(config, model, dl, opt, scheduler, epoch):
    model.train()
    model = model.cuda()
    for ep in tqdm(range(epoch)):
        model.train()
        model = model.cuda()
        # pbar = tqdm(dl)
        for i, batch in enumerate(dl):
            x, y = batch[0].cuda(), batch[1].cuda()
            out = model(x)
            loss = F.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if scheduler is not None:
            scheduler.step(ep)
        if ep % 10 == 9:
            file = open('./results.txt',mode='a+')
            acc = test(model, test_dl)
            if acc > config['best_acc']:
                config['best_acc'] = acc
                save(config['method'], config['name'], model, acc, ep)
            if ep ==99:
                file.write(" acc: {:.3f}     \n".format(acc))
            else:
                file.write(" acc: {:.3f}     ".format(acc))
    model = model.cpu()
    return model


@torch.no_grad()
def test(model, dl):
    model.eval()
    acc = Accuracy()
    #pbar = tqdm(dl)
    model = model.cuda()
    for batch in dl:  # pbar:
        x, y = batch[0].cuda(), batch[1].cuda()
        out = model(x).data
        acc.update(out.argmax(dim=1).view(-1), y, 0)

    return acc.result()[0]


if __name__ == '__main__':
    torch.cuda.set_device(0)
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--model', type=str, default='vit_base_patch16_224_in21k')
    parser.add_argument('--dataset', type=str, default='dtd')
    parser.add_argument('--method', type=str, default='MetaAdapter')
    parser.add_argument('--adapter', type=str, default='ConvAdapter') # ConvAdapter AttnAdapter MLPAdapter
    args = parser.parse_args()
    print(args)
    set_seed(args.seed)
    config = get_config(args.method, args.dataset)
    model = create_model(args.model, checkpoint_path='/path/to/ViT-B_16.npz', drop_path_rate=0.1)
    if args.dataset in ["cub", "nabirds", "stanford_cars", "stanford_dogs", "fgvc_flowers"]: 
        train_dl, test_dl = get_fgvc_data(args.dataset)
    else:
        train_dl, test_dl = get_data(args.dataset)

    set_MetaAdapter(model=model, method=args.method, adapter=args.adapter, dim=16, s=config['scale'], xavier_init=config['xavier_init'])

    trainable = []
    model.reset_classifier(config['class_num'])
    
    config['best_acc'] = 0
    config['method'] = args.method
    
    for n, p in model.named_parameters():
        print("name: ",n)
        if 'metaAdapter' in n or 'head' in n or 'norm0' in n:
            trainable.append(p)
        else:
            p.requires_grad = False

    n_parameters = sum(p.numel() for p in model.parameters() )
    print('number of total params (M): %.2f' % (n_parameters))
    n1_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of tuneable params (M): %.2f,ratio is %f' % (n1_parameters, n1_parameters/n_parameters))


    opt = AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    scheduler = CosineLRScheduler(opt, t_initial=100,
                                  warmup_t=10, lr_min=1e-5, warmup_lr_init=1e-6, decay_rate=0.1)
    model = train(config, model, train_dl, opt, scheduler, epoch=100)
    print(config['best_acc'])
