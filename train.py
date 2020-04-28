import argparse, os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
import torch.backends.cudnn as cudnn

import pickle
import numpy as np
from utils import Vocabulary, Custom_Flickr30k, collate_fn
from models import EncoderCNN, DecoderRNNwithAttention
from BLEU import bleu_eval


def get_parser():
    parser = argparse.ArgumentParser(description='Flickr30k Training')
    parser.add_argument('-batch_size', type=int, default=512,
                        help='batch size')
    parser.add_argument('-embed_size', type=int, default=256,
                        help='embedding demension size')
    parser.add_argument('-hid_size', type=int, default=512,
                        help='hidden demension size')
    parser.add_argument('-attn_size', type=int, default=512,
                        help='attention demension size')
    parser.add_argument('-drop', type=float, default=0.5,
                        help='dropout percentage')
    parser.add_argument('-epoch', type=int, default=300,
                        help='number of training epochs')
    parser.add_argument('-fine_tune', type=bool, default=True,
                        help='whether to fine-tune the encoder or not')
    parser.add_argument('-lr', type=float, default=0.01,
                        help='initial learning rate')
    parser.add_argument('--save', type=str, default=Path.cwd(),
                        help='directory to save logs and models.')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    return parser


def main():

    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = get_parser().parse_args()

    NUM_WORKERS = 4
    CROP_SIZE = 256
    NUM_PIXELS = 64
    ENCODER_SIZE = 2048
    learning_rate = args.lr
    start_epoch = 0

    min_BLEU = 1_000_000

    vocab = pickle.load(open('vocab.p', 'rb'))

    train_transform = transforms.Compose([
            transforms.RandomCrop(CROP_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.444, 0.421, 0.385),
                                 (0.285, 0.277, 0.286))])

    val_transform = transforms.Compose([
            transforms.CenterCrop(CROP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.444, 0.421, 0.385),
                                 (0.285, 0.277, 0.286))])

    train_loader = torch.utils.data.DataLoader(
            dataset=Custom_Flickr30k('../flickr30k-images','../flickr30k-captions/results_20130124.token', vocab, transform=train_transform, train=True),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(
            dataset=Custom_Flickr30k('../flickr30k-images','../flickr30k-captions/results_20130124.token', vocab, transform=val_transform, train=False),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn)

    # Initialize models
    encoder = EncoderCNN(args.fine_tune).to(device)
    decoder = DecoderRNNwithAttention(len(vocab), args.embed_size, args.hid_size, 1, args.attn_size, ENCODER_SIZE, NUM_PIXELS, dropout=args.drop).to(device)

    # Initialize optimization
    criterion = torch.nn.CrossEntropyLoss()
    if args.fine_tune:
        params = list(encoder.parameters()) + list(decoder.parameters())
    else:
        params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            min_BLEU = checkpoint['min_BLEU']
            encoder.load_state_dict(checkpoint['encoder'])
            decoder.load_state_dict(checkpoint['decoder'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else: print("No checkpoint found at '{}'".format(args.resume))

    XEntropy = AverageMeter()
    PPL = AverageMeter()

    # Save
    file = open(f'{args.save}/resuts.txt','a')
    file.write('Loss,PPL,BLEU \n')
    file.close()

    for epoch in range(start_epoch, args.epoch):
        print('Epoch {}'.format(epoch+1))
        print('training...')
        for i, (images, captions, lengths) in enumerate(train_loader):
            # Batch to device
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            encoder.train()
            decoder.train()

            features = encoder(images)
            predictions, attention_weights = decoder(features, captions, lengths)

            scores = pack_padded_sequence(predictions[:,:-1,:], torch.tensor(lengths)-2, batch_first=True).cpu()
            targets = pack_padded_sequence(captions[:,1:-1], torch.tensor(lengths)-2, batch_first=True).cpu()

            loss = criterion(scores.data, targets.data)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            XEntropy.update(loss.item(), len(lengths))
            PPL.update(np.exp(loss.item()), len(lengths))
        print('Train Perplexity = {}'.format(PPL.avg))

        if epoch % 50 == 0:
            learning_rate /= 5
            for param_group in optimizer.param_groups: param_group['lr'] = learning_rate

        print('validating...')
        curr_BLEU = bleu_eval(encoder, decoder, val_loader, args.batch_size)
        is_best = curr_BLEU < min_BLEU
        min_BLEU = max(curr_BLEU, min_BLEU)
        save_checkpoint({
            'epoch': epoch + 1, 'encoder': encoder.state_dict(), 'decoder': decoder.state_dict(),
            'min_BLEU': min_BLEU, 'optimizer' : optimizer.state_dict(),
        }, is_best)

        print('Validation BLEU = {}'.format(curr_BLEU))

        # Save
        file = open(f'{args.save}/resuts.txt','a')
        file.write('{},{},{} \n'.format(XEntropy.avg,PPL.avg,curr_BLEU))
        file.close()

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


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f'{args.save_dir}/model_best.pth.tar')


if __name__ == '__main__': main()
