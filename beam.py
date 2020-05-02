from torchvision import transforms

class BeamNode(object):
    def __init__(self, word_ids, scores, seq):
        '''
        :param wordIds:  Tensor, shape=(batch_size, 1), dtype=long
        :param scores:   Tensor, shape=(batch_size, 1), dtype=float
        :param seq:      List,   shape=(batch_size, length)
        '''
        self.word_ids = word_ids
        self.scores = scores
        self.seq = seq
        self.imgs = None
        self.targets = None
    
    def add_imgs(self, imgs):
        self.imgs = imgs
    
    def add_targets(self, targets):
        self.targets = targets
        
def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__, reverse=True)

def print_beam_outputs(beams, vocab, num_to_print = 10):
    inv_transform = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                            std = [1/0.285, 1/0.277, 1/0.286]),
                                        transforms.Normalize(mean = [-0.444, -0.421, -0.385],
                                                             std = [1., 1., 1.]),
                                        transforms.ToPILImage(),
                                       ]) # converts transformed image back to PIL
    
    for i in range(min(len(beams[0].seq), num_to_print)):
        sorted_beam_idxs = argsort([beams[k].scores[i] for k in range(len(beams))]) # sort the outputs in descending score order
        display(inv_transform(beams[0].imgs[i])) # display the image
        
        captions = [[]]
        idx = 0
        for tok in beams[0].targets[i]:
            if tok.item() == 4: # '<break>'
                idx += 1
                captions.append([])
            elif tok.item() != 0: # '<pad>':
                captions[idx].append(tok.item())
        
        print('Target Captions:')
        for caption in captions:
            print(' '.join([vocab.idx2word[tok] for tok in caption]))
        print('')
        print('Predicted Captions:')
        for idx in sorted_beam_idxs:
            print(' '.join([vocab.idx2word[tok] for tok in beams[idx].seq[i]]))
        