import math
import nltk
import numpy as np
import torch

def bleu_eval(encoder, decoder, data_loader, batch_size, device, beam_size=None):
    with torch.no_grad():
        true_outputs = [] # e.g: [[ref1_1, ref1_2], [ref2_1, ...], ....]
        decoder_outputs = [] # e.g: [out1, out2, out3]
        for i, (images, captions, lengths) in enumerate(data_loader):
            for caption in captions:
                caption = caption.numpy().astype(str).tolist()
                idx = 0
                curr_img_captions = [[]] # e.g: [ref1_1, ref1_2, ...]
                for tok in caption[1:-1]:
                    if tok == '4': # '<break>'
                        idx += 1
                        curr_img_captions.append([])
                    elif tok != '0': # '<pad>'
                        curr_img_captions[idx].append(tok)
                true_outputs.append(curr_img_captions)

            images = images.to(device)
            captions = captions.to(device)

            features = encoder(images)
            max_sequence_length = int(np.array(captions != 0).sum(axis = 1).mean()+1)
            if beam_size is None: # greedy
            	sample = decoder.sample(features, max_seq_length=max_sequence_length).cpu()
            	decoder_outputs.extend(sample.numpy().astype(str).tolist())
            else:
                beams = decoder.beam_sample(features, 
                    imgs=images, 
                    targets=captions,
                    beam_size=beam_size,
                    max_seq_length=max_sequence_length)
                for i in range(batch_size):
                    best_score_idx = -1
                    best_score = 0
                    for j in range(len(beams)):
                        if beams[j].scores[i] > best_score:
                            best_score_idx = j
                            best_score = beams[j].scores[i]
                    decoder_outputs.append(beams[best_score_idx].seq[i])

        predictions = []
        for pred in decoder_outputs:
            curr = []
            for tok in pred:
                if tok == '2': # '<end>'
                    break
                curr.append(tok)
            predictions.append(curr)
        return predictions, nltk.bleu_score.corpus_bleu(true_outputs, predictions, weights=(0.33, 0.33, 0.33, 0.0))