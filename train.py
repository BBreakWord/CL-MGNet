import argparse
import auxil
import os
from dataset import *
import torch.nn as nn
import torch
import torch.nn.parallel
from torch.utils.data.dataset import Dataset
import models.models as clmgnet
import numpy as np
from auxil import str2bool
import matplotlib.pyplot as plt


def load_hyper(args):
    # load dataset
    data, label, numclass = auxil.loadData(args.dataset, num_components=args.components)
    data_shape = data.shape

    # padding
    PATCH_LENGTH = int((args.spatialsize - 1) // 2)
    padded_data = np.lib.pad(data, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)), 'constant', constant_values=0)
    bands = data_shape[-1]
    numberofclass = np.max(label)
    print("bands: {}, numofclass: {}".format(bands, numberofclass))
    labels = label.reshape((label.shape[0] * label.shape[1]))
    labels = labels.astype(np.int32)

    # split train-test
    train_indices, test_indices = auxil.sampling(1 - args.tr_percent, labels)
    _, total_indices = auxil.sampling(1, labels)
    total_size, train_size, test_size = len(total_indices), len(train_indices), len(test_indices)
    x_all, x_train, x_test, y_all, y_train, y_test = auxil.generate_data(train_size, train_indices, test_size, test_indices, total_size, total_indices,
             data , PATCH_LENGTH, padded_data, bands, labels)
    print("total_size: {}, train_size: {}, test_size: {}\ntotal_data_shape: {}, train_data_shape: {}, test_data_shape: {}".format(
        total_size, train_size, test_size, x_all.shape, x_train.shape, x_test.shape
    ))

    # del pixels, labels
    train_hyper = HSIDataset((np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32), y_train), None)
    test_hyper = HSIDataset((np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32), y_test), None)
    full_hyper = HSIDataset((np.transpose(x_all, (0, 3, 1, 2)).astype(np.float32), y_all), None)
    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_hyper, batch_size=args.tr_bsize, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_hyper, batch_size=args.te_bsize, shuffle=False, **kwargs)
    full_loader = torch.utils.data.DataLoader(full_hyper, batch_size=args.te_bsize, shuffle=False, **kwargs)
    return labels, full_loader, train_loader, test_loader, numberofclass, bands, data_shape, total_indices


def train(trainloader, model, ce_criterion, cc_criterion, optimizer, optimizer_cc, epoch, use_cuda, args):
    model.train()
    accs   = np.ones((len(trainloader))) * -1000.0
    losses = np.ones((len(trainloader))) * -1000.0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        print('\repoch:'+str(epoch + 1)+'  |  progress: '+str(batch_idx + 1)+'/'+str(len(trainloader)), end=" ")
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda().long()
        outputs, band_weights = model(inputs)
        loss1 = ce_criterion(outputs, targets)
        loss = loss1
        loss2 = cc_criterion(band_weights, targets)
        loss += args.mu * loss2
        losses[batch_idx] = loss.item()
        accs[batch_idx] = auxil.accuracy(outputs.data, targets.data)[0].item()
        optimizer.zero_grad()
        optimizer_cc.zero_grad()
        loss.backward()
        for param in cc_criterion.parameters():
            param.grad.data *= (1. / args.mu)
            optimizer_cc.step()
        optimizer.step()
        CenterClipper(cc_criterion)
        # print('next iteration!')
    print('\n')
    return (np.average(losses), np.average(accs))


def test(testloader, model, ce_criterion, cc_criterion, epoch, use_cuda, args):
    model.eval()
    accs   = np.ones((len(testloader))) * -1000.0
    losses = np.ones((len(testloader))) * -1000.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda().long()
            outputs, band_weights = model(inputs)
            losses[batch_idx] = ce_criterion(outputs, targets).item()
            losses[batch_idx] += args.mu * cc_criterion(band_weights, targets).item()
            accs[batch_idx] = auxil.accuracy(outputs.data, targets.data, topk=(1,))[0].item()
    return (np.average(losses), np.average(accs))


def predict(testloader, model, use_cuda):
    model.eval()
    predicted = []
    with torch.no_grad():	
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda: inputs = inputs.cuda()
            pred, weights = model(inputs)
            predicted.extend(pred.cpu().numpy())
    return np.array(predicted)


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def CenterClipper(cc_criterion):
    cc_criterion.weightcenters.data.clamp_(0, 1)


class CategoryConsistencyLoss(nn.Module):
    def __init__(self, num_classes, embedding_size):
        super(CategoryConsistencyLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.weightcenters = nn.Parameter(torch.normal(0, 1, (num_classes, embedding_size)))

    def forward(self, x, labels):
        if len(x.size()) == 1:
            x = x.unsqueeze(0)
        batch_size = x.size(0)
        dist_metric = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                      torch.pow(self.weightcenters, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        dist_metric.addmm_(x, self.weightcenters.t(), beta=1, alpha=-2)

        dist = dist_metric[range(batch_size), labels]
        loss = dist.clamp(1e-12, 1e+12).sum() / batch_size
        return loss


def main():
    parser = argparse.ArgumentParser(description='Configuration')
    parser.add_argument('--epochs', default=500, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--components', default=None, type=int, help='dimensionality reduction')
    parser.add_argument('--dataset', default='IP', type=str, help='dataset (options: IP, PU, Houston2013, Dioni, Houston2018)')
    parser.add_argument('--tr_percent', default=0.10, type=float, help='samples of train set')
    parser.add_argument('--tr_bsize', default=32, type=int, help='mini-batch train size (default: 32)')
    parser.add_argument('--te_bsize', default=32, type=int, help='mini-batch test size (default: 32)')
    parser.add_argument('--inplanes', dest='inplanes', default=256, type=int, help='feature dims')
    parser.add_argument('--num_blocks', dest='num_blocks', default=4, type=int, help='number of EPSA Resblock')
    parser.add_argument('--num_heads', dest='num_heads', default=4, type=int, help='number of MHEA head')
    parser.add_argument('--num_encoders', dest='num_encoders', default=1, type=int, help='number of attention-MLP encoder')
    parser.add_argument('--spatialsize', dest='spatialsize', default=13, type=int, help='spatial-spectral patch dimension')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, help='weight decay for swl (default: 1e-4)')
    parser.add_argument('--mu', type=float, default=10., help='weight of category consistency loss')
    parser.add_argument('--resume', type=str2bool, default='false')

    parser.set_defaults(bottleneck=True)
    
    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}

    patchesLabels, full_loader, train_loader, test_loader, num_classes, n_bands, data_shape, total_indices = load_hyper(args)
    print('[i] Dataset finished!')
    # Use CUDA
    use_cuda = torch.cuda.is_available()
    if use_cuda: torch.backends.cudnn.benchmark = True

    # set model
    model_name = "CL-MGNet_PS-{}_mu-{}".format(args.spatialsize, args.mu)
    model = clmgnet.CLMGNet(num_classes, n_bands, args.spatialsize, args.inplanes, num_blocks=args.num_blocks, num_heads=args.num_heads, num_encoders=args.num_encoders)
    if use_cuda: model = model.cuda()
    ce_criterion = torch.nn.CrossEntropyLoss()
    cc_criterion = CategoryConsistencyLoss(num_classes=num_classes, embedding_size=n_bands)
    if use_cuda:
        cc_criterion = cc_criterion.cuda()
        ce_criterion = ce_criterion.cuda()
    paras = dict(model.named_parameters())
    paras_group = []
    for k, v in paras.items():
        if 'swl' in k:
            paras_group += [{'params': [v], 'weight_decay': args.weight_decay}]
        else:
            paras_group += [{'params': [v], 'weight_decay': 1e-4}]
    optimizer = torch.optim.SGD(paras_group, args.lr,
                                momentum=args.momentum, nesterov=True)
    categotyconsistencyloss_paras_group = [{'params': cc_criterion.parameters(), 'weight_decay': args.weight_decay}]
    optimizer_cc = torch.optim.SGD(categotyconsistencyloss_paras_group, args.lr,
                                momentum=args.momentum, nesterov=True)

    # training
    best_acc = -1
    init_epoch = 0
    if args.resume:
        checkpoint = torch.load('current_checkpoints/' + str(args.dataset) + '_tr-' + str(args.tr_percent) + '_' + model_name + '.pth')
        init_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []
    for epoch in range(init_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        train_loss, train_acc = train(train_loader, model, ce_criterion, cc_criterion, optimizer, optimizer_cc, epoch, use_cuda, args)
        with torch.no_grad():
            test_loss, test_acc = test(test_loader, model, ce_criterion, cc_criterion, epoch, use_cuda, args)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        print("EPOCH", epoch + 1, "Train Loss", train_loss, "Train Accuracy", train_acc, end=', ')
        print("Test Loss", test_loss, "Test Accuracy", test_acc)
        # save model
        torch.save(state, 'current_checkpoints/' + str(args.dataset) + '_tr-' + str(args.tr_percent) + '_' + model_name + '.pth')
        if test_acc > best_acc:
            state = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'optimizer_cc': optimizer_cc.state_dict()
            }
            torch.save(state, 'best_checkpoints/' + str(args.dataset) + '_tr-' + str(args.tr_percent) + '_' + model_name + '.pth')
            best_acc = test_acc

    # test
    checkpoint = torch.load('current_checkpoints/' + str(args.dataset) + '_tr-' + str(args.tr_percent) + '_' + model_name + '.pth')
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    
    with torch.no_grad():
        test_loss, test_acc = test(test_loader, model, ce_criterion, cc_criterion, start_epoch, use_cuda, args)
    pred = predict(full_loader, model, use_cuda)
    prediction = np.argmax(pred, axis=1)
    print("FINAL:      LOSS", test_loss, "ACCURACY", test_acc)

    # save classification map
    de_map = np.zeros(patchesLabels.shape, dtype=np.int32)
    de_map[total_indices] = prediction + 1
    de_map = np.reshape(de_map, (data_shape[0], data_shape[1]))
    w, h = de_map.shape
    plt.figure(figsize=[h/100.0, w/100.0])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
    plt.axis('off')
    plt.imshow(de_map, cmap='jet')
    plt.savefig(os.path.join('classification_map/' + str(args.dataset) + '_tr-' + str(args.tr_percent) + '_' + model_name + '_classification-map_' + 'OA' + str(test_acc) + '.png'), format='png')
    plt.close()

    # save results
    classification, confusion, results = auxil.reports(np.argmax(predict(test_loader, model, use_cuda), axis=1), np.array(test_loader.dataset.__labels__()), args.dataset)
    print(args.dataset, results)
    str_res = np.array2string(np.array(results), max_line_width=200)
    print(str_res)
    log = ('Dataset = %s, patch size = %d, tr_percent = %.2f, mu = %.4f, Loss = %.8f, Accuracy = %4.4f\nResults = %s\n') % (args.dataset, args.spatialsize, args.tr_percent, args.mu, test_loss, test_acc, str_res)
    with open(os.path.join('results', model_name + '.txt'), 'a') as f:
        f.write(log)


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()

