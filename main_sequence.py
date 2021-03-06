from __future__ import print_function

import argparse
import csv
import os
import os.path
import shutil
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
#import torchvision.models as models
#import torchvision.transforms as transforms
#from PIL import Image
from models.single_modality.word_embeddings import load_data as w_load_data
from models.single_modality.word_embeddings import seq_model as w_seq_model


model_names = []
model_names.append('words_embeddings')

parser = argparse.ArgumentParser(description='PyTorch Cats vs Dogs fine-tuning example')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument(
    '--arch',
    metavar='ARCH',
    default='words_embeddings',
    choices=model_names,
    help='model architecture: ' + ' | '.join(model_names) + ' (default: words_embeddings)')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=1, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--resume2', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--test', dest='test', action='store_true', help='evaluate model on test set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--cweights', default='', type=str, metavar='PATH', help='path to the file containing the list of labels')
parser.add_argument('--softbatch', default=18, type=int, metavar='N', help='optimize parameters afer softbatch/batch_size samples')


best_prec1 = 0


USE_CUDA=True


def main():
    global args, best_prec1
    args = parser.parse_args()

    if args.arch == "words_embeddings":
        load_data = w_load_data
        
    # Data loading code
    train_loader = data.DataLoader(
                    load_data.TranscriptionsReader('/mnt/3T-NAS/Databases/jobScreening_cvpr17/train/transcription_training.pkl',
                                                    '/mnt/3T-NAS/Databases/jobScreening_cvpr17/train/annotation_training.pkl'), 
                        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, 
                        collate_fn = load_data.TranscriptionsReader.my_collate
                    )


    train_loader = data.DataLoader(
                    load_data.TranscriptionsReader('/mnt/3T-NAS/Databases/jobScreening_cvpr17/train/transcription_training.pkl',
                                                    '/mnt/3T-NAS/Databases/jobScreening_cvpr17/train/annotation_training.pkl'), 
                        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, 
                        collate_fn = load_data.TranscriptionsReader.my_collate
                    )

    val_loader = data.DataLoader(
                    load_data.TranscriptionsReader('/mnt/3T-NAS/Databases/jobScreening_cvpr17/train/transcription_training.pkl',
                                                    '/mnt/3T-NAS/Databases/jobScreening_cvpr17/train/annotation_training.pkl'), 
                        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, 
                        collate_fn = load_data.TranscriptionsReader.my_collate
                    )

    test_loader = data.DataLoader(
                    load_data.TranscriptionsReader('/mnt/3T-NAS/Databases/jobScreening_cvpr17/train/transcription_training.pkl',
                                                    '/mnt/3T-NAS/Databases/jobScreening_cvpr17/train/annotation_training.pkl'), 
                        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, 
                        collate_fn = load_data.TranscriptionsReader.my_collate
                    )




    if args.arch == "words_embeddings":
        model = w_seq_model.Word_Embeddings_sequence_model(vocab_size=len(train_loader.dataset.word2id),embedding_size=32,
            nhid=256, nlayers=2)

    #model = torch.nn.DataParallel(model).cuda()
    if USE_CUDA:
        model = model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    if args.test:
        print("Testing the model and generating a output csv for submission")
        test(test_loader, train_loader.dataset.class_to_idx, model)
        return
    # define loss function (criterion) and pptimizer
    if args.cweights != '':
        from sklearn.utils import class_weight
        clabels = []
        with open(args.cweights, 'r') as stream:
            for line in stream:
                line=line.strip()
                clabels.append(line)
        class_weight = class_weight.compute_class_weight('balanced', np.unique(clabels),clabels)
        class_weight = torch.from_numpy(class_weight).float()
    else:
        class_weight = None

    #criterion = nn.CrossEntropyLoss(weight=class_weight)
    criterion = nn.MSELoss()
    if USE_CUDA:
        criterion.cuda()

    #optimizer = optim.Adam(model.module.fc.parameters(), args.lr, weight_decay=args.weight_decay)
    optimizer = optim.Adam( filter(lambda p: p.requires_grad, model.parameters()) , args.lr, weight_decay=args.weight_decay)
    

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best Accuracy and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = {}
    for key in train_loader.dataset.traits:
        accs[key] = AverageMeter()
    
    #import tracemalloc
    #tracemalloc.start()
    # switch to train mode
    model.train()
    
    soft_batch = int(np.ceil(args.softbatch/args.batch_size))
    #acc_loss = 0
    optimizer.zero_grad()

    
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if USE_CUDA:
            target = target.cuda(async=True)
        #print(images)
        #image_var = torch.autograd.Variable(images)
        image_var = images
        #print(images.size()[0])
        label_var = torch.autograd.Variable(target)
        hidden = model.init_hidden(images.size()[0])
        if USE_CUDA:
            hidden = (hidden[0].cuda(), hidden[1].cuda())
        #print(label_var)
        #label_var + "hola"

        # compute y_pred
        y_pred = model(image_var, hidden)
        #print(y_pred)
        loss = criterion(y_pred, label_var)

        # measure accuracy and record loss
        #prec1, prec1 = accuracy(y_pred.data, target, topk=(1, 1))
        losses.update(loss.data[0], images.size(0))
        #acc.update(prec1[0], images.size(0))
        curr_accuracy = regression_accuracy(y_pred.data, target).cpu().numpy()[0]
        for i, key in enumerate(train_loader.dataset.traits):
            accs[key].update(curr_accuracy[i], images.size(0))

        # compute gradient and do SGD step
        #optimizer.zero_grad()
        loss.backward()
        #optimizer.step()
        
        if (i+1) % soft_batch == 0:
            print("Optimizing parameters")
            optimizer.step()
            acc_loss = 0
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        #snapshot = tracemalloc.take_snapshot()
        #stats_file = open('stats.txt','w')
        #top_stats = snapshot.statistics('lineno')
        #for ts in top_stats:
            #stats_file.write(str(ts))
            #stats_file.write('\n')
        #stats_file.close()
        #return

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Interview Acc {interview.val:.3f} ({interview.avg:.3f})\t'
                  'agreeableness {agreeableness.val:.3f} ({agreeableness.avg:.3f})\t'
                  'conscientiousness {conscientiousness.val:.3f} ({conscientiousness.avg:.3f})\t'
                  'extraversion {extraversion.val:.3f} ({extraversion.avg:.3f})\t'
                  'neuroticism {neuroticism.val:.3f} ({neuroticism.avg:.3f})\t'
                  'openness {openness.val:.3f} ({openness.avg:.3f})'
                  .format(
                      epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, interview=accs['interview'], agreeableness=accs['agreeableness'], conscientiousness=accs['conscientiousness'], extraversion=accs['extraversion'], neuroticism=accs['neuroticism'], openness=accs['openness'] ))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (images, labels) in enumerate(val_loader):
        
        if USE_CUDA:
            labels = labels.cuda(async=True)
        #print(images)
        #image_var = torch.autograd.Variable(images)
        image_var = images
        label_var = torch.autograd.Variable(labels, volatile=True)
        hidden = model.init_hidden(image_var.size()[0])
        if USE_CUDA:
            #hidden = hidden.cuda()
            hidden = (hidden[0].cuda(), hidden[1].cuda())

        #print(label_var)
        #label_var + "hola"

        # compute y_pred
        y_pred = model(image_var, hidden)
        loss = criterion(y_pred, label_var)

        # measure accuracy and record loss
        #prec1, temp_var = accuracy(y_pred.data, labels, topk=(1, 1))
        losses.update(loss.data[0], images.size(0))
        #acc.update(prec1[0], images.size(0))
        acc.update( 0.5, images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('TrainVal: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses, acc=acc))

    print(' * Accuracy {acc.avg:.3f}'.format(acc=acc))

    return acc.avg


def test(test_loader, class_to_idx, model):
    csv_map = {}
    csv2_map = {}
    
    # switch to evaluate mode
    model.eval()
    old_sbj_id = ""
    acc_probs = np.zeros(7)
    ncount = 0
    for i, (images, filepath) in enumerate(test_loader):
        # pop extension, treat as id to map
        #filepath = os.path.splitext(os.path.basename(filepath[0]))[0]
        #filepath = int(filepath)

        image_var = torch.autograd.Variable(images, volatile=True)
        y_pred = model(image_var)
        # get the index of the max log-probability
        smax = nn.Softmax()
        smax_out = smax(y_pred)[0]
        
        angry_prob = smax_out.data[class_to_idx['Angry']]
        disgust_prob = smax_out.data[class_to_idx['Disgust']]
        fear_prob = smax_out.data[class_to_idx['Fear']]
        happy_prob = smax_out.data[class_to_idx['Happy']]
        neutral_prob = smax_out.data[class_to_idx['Neutral']]
        sad_prob = smax_out.data[class_to_idx['Sad']]
        surprise_prob = smax_out.data[class_to_idx['Surprise']]
                
        
        #cat_prob = smax_out.data[0]
        #dog_prob = smax_out.data[1]
        #prob = dog_prob
        #if cat_prob > dog_prob:
            #prob = 1 - cat_prob
        #prob = np.around(prob, decimals=4)
        #prob = np.clip(prob, .0001, .999)
        csv_map[filepath] = [angry_prob, disgust_prob, fear_prob, happy_prob, neutral_prob, sad_prob, surprise_prob]
        #print(filepath, {"Angry" : angry_prob, "Disgust" : disgust_prob, "Fear" : fear_prob, "Happy": happy_prob, "Neutral" : neutral_prob, "Sad" : sad_prob, "Surprise" : surprise_prob})

        sbj_id = str(filepath).strip().split('/')[-1].split('_')[0]
        if sbj_id != old_sbj_id:
            if old_sbj_id != "":
                acc_probs = acc_probs / ncount
                
                csv2_map[old_sbj_id] = acc_probs.tolist()
                print(old_sbj_id, acc_probs.tolist())
            acc_probs = np.array([angry_prob, disgust_prob, fear_prob, happy_prob, neutral_prob, sad_prob, surprise_prob])
            ncount = 1
        else:
            acc_probs = acc_probs + np.array([angry_prob, disgust_prob, fear_prob, happy_prob, neutral_prob, sad_prob, surprise_prob])
            ncount = ncount +1
            
        old_sbj_id = sbj_id
            
    with open(os.path.join(args.data, 'entry2.csv'), 'w') as csvfile:
        csv_w = csv.writer(csvfile)
        csv_w.writerow(('id', 'Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'))
        for row in sorted(csv2_map.items()):
            csv_w.writerow( tuple((str(row[0])+','+str(','.join([str(a) for a in row[1]]))).split(',')) )


    with open(os.path.join(args.data, 'entry.csv'), 'w') as csvfile:
        csv_w = csv.writer(csvfile)
        csv_w.writerow(('id', 'Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'))
        for row in sorted(csv_map.items()):
            csv_w.writerow( tuple((str(row[0])+','+str(','.join([str(a) for a in row[1]]))).split(',')) )


    return


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


def accuracy(y_pred, y_actual, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = y_actual.size(0)

    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_actual.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res
  
def regression_accuracy(y_pred, y_actual):
    return torch.mean(1.0 - torch.abs(y_pred-y_actual), dim=0)


class TestImageFolder(data.Dataset):
    def __init__(self, root, transform=None):
        #images = []
        #for filename in os.listdir(root):
            #if filename.endswith('jpg'):
                #images.append('{}'.format(filename))

        images = []
        for target in sorted(os.listdir(root)):
            d = os.path.join(root, target)
            if not os.path.isdir(d):
                continue

            for r, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if fname.endswith('jpg'):
                        path = os.path.join(r, fname)
                        images.append('{}'.format(path))

        self.root = root
        self.imgs = images
        self.transform = transform

    def __getitem__(self, index):
        filename = self.imgs[index]
        img = Image.open(filename)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, filename

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    main()
