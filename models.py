import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from math import sqrt
from beam import BeamNode
import spacy
import numpy as np


class SimpleEncoderCNN(nn.Module):
    def __init__(self, embed_size, fine_tune=False):
        super(SimpleEncoderCNN, self).__init__()
        self.fine_tune = fine_tune
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.1)

    def forward(self, images):
        if self.fine_tune:
            features = self.resnet(images)
        else:
            with torch.no_grad():
                features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class SimpleDecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(SimpleDecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None, max_seq_length=20):
        """Gready Search"""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True) # switch from resnet50 to resnet152?
        modules = list(resnet.children())[:-2]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)

    def forward(self, images):
        features = self.resnet(images)

        return features

class AttentionLayer(nn.Module):

    def __init__(self, encoder_size, hidden_size, attention_size):
        super().__init__()
        # Arguments
        self.encoder_size = encoder_size
        self.hidden_size = hidden_size
        self.attention_size = attention_size

        self.encoder_transform = nn.Linear(self.encoder_size, self.attention_size)
        self.decoder_transform = nn.Linear(self.hidden_size, self.attention_size)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(attention_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, decoder_hidden):

        encoder_attn = self.encoder_transform (features)
        decoder_attn = self.decoder_transform(decoder_hidden)
        similarity_score = encoder_attn + decoder_attn.unsqueeze(1)
        attention_scores = self.linear(self.relu(similarity_score))
        attention_weights = self.softmax(attention_scores.squeeze(2))
        output = (features * attention_weights.unsqueeze(2)).sum(dim=1)

        return output, attention_weights #(batch_size, encoder_size), (batch_size, number_of pixels)

class DecoderRNNwithAttention(nn.Module):
    def __init__(self, vocab, hidden_size, num_layers, attention_size, encoder_size=2048, num_pixels = 64, dropout=0.5):
        super().__init__()
        # Arguments
        self.vocab_size = len(vocab)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention_size = attention_size
        self.encoder_size = encoder_size
        self.num_pixels = num_pixels
        self.dropout = nn.Dropout(p=dropout)

        nlp = spacy.load('en_core_web_md') # very slow
        _, self.embed_size = nlp.vocab.vectors.shape
        #self.embed_size = 256
        self.embed = nn.Embedding(self.vocab_size, self.embed_size, 0)
        pretrained_weight = np.array(list(map(lambda x: nlp(x).vector, vocab.all_words))) # pretrained_weight is a numpy matrix of shape (num_embeddings, embedding_dim)
        self.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))

        self.pool = nn.AvgPool2d(int(sqrt(num_pixels)))
        self.init_hidden = nn.Linear(self.encoder_size, self.hidden_size)
        self.init_memory = nn.Linear(self.encoder_size, self.hidden_size)

        # self.lstm = nn.LSTM(self.embed_size + self.encoder_size,
        #             self.hidden_size, self.num_layers,
        #             batch_first=True,
        #             dropout=dropout if num_layers > 1 else 0,
        #             )

        self.lstm = nn.LSTMCell(self.embed_size + self.encoder_size,self.hidden_size)

        self.attention = AttentionLayer(self.encoder_size, self.hidden_size, self.attention_size)
        self.beta = nn.Linear(self.hidden_size, self.encoder_size)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, features, captions, lengths):

        batch_size = captions.size(0)
        target_length = max(lengths)-1

        init_features = self.pool(features).squeeze()
        hidden = self.init_hidden(init_features)
        memory = self.init_memory(init_features)

        features = features.permute(0, 2, 3, 1)
        features  = features.view(batch_size, -1, self.encoder_size)

        embeddings = self.embed(captions)

        predictions = torch.zeros(batch_size, target_length, self.vocab_size).to(self.embed.weight.device)
        attention_weights = torch.zeros(batch_size, target_length, self.num_pixels).to(self.embed.weight.device)

        for i in range(target_length-1):
            batch_subset = sum([l > i for l in lengths])
            attention_weighted_encoding, attention_weight = self.attention(features[:batch_subset], hidden[:batch_subset])
            gate = self.sigmoid(self.beta(hidden[:batch_subset]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            hidden, memory = self.lstm(
                torch.cat([embeddings[:batch_subset, i, :], attention_weighted_encoding], dim=1),
                (hidden[:batch_subset], memory[:batch_subset]))  # (batch_size_t, decoder_dim)
            preds = self.out(self.dropout(hidden))
            predictions[:batch_subset, i, :] = preds
            attention_weights[:batch_subset, i, :] = attention_weight

        return predictions, attention_weights

    def sample(self, features, max_seq_length=20, return_attention = False):
        """Gready Search"""
        with torch.no_grad():

            batch_size = features.size(0)

            init_features = self.pool(features).squeeze()
            hidden = self.init_hidden(init_features)
            memory = self.init_memory(init_features)

            features = features.permute(0, 2, 3, 1)
            features  = features.view(batch_size, -1, self.encoder_size)

            predictions = torch.Tensor([1]).long().to(self.embed.weight.device).expand(batch_size, 1).long()
            attention_weights = torch.zeros(batch_size, max_seq_length, self.num_pixels).to(self.embed.weight.device)

            for i in range(max_seq_length):
                embeddings = self.embed(predictions[:,i])
                attention_weighted_encoding, attention_weight = self.attention(features, hidden)
                gate = self.sigmoid(self.beta(hidden))
                attention_weighted_encoding = gate * attention_weighted_encoding
                hidden, memory = self.lstm(
                    torch.cat([embeddings, attention_weighted_encoding], dim=1),
                    (hidden, memory))  # (batch_size_t, decoder_dim)
                out = self.out(self.dropout(hidden))
                _, preds = out.max(1)
                predictions = torch.cat((predictions, preds.unsqueeze(1)), 1)
                attention_weights[:, i, :] = attention_weight

            if return_attention:
                return predictions, attention_weights
            else:
                return predictions

    def beam_sample(self, features, targets=None, imgs=None, beam_size=3, max_seq_length=20, return_attention = False):
        """Beam Search"""
        with torch.no_grad():
            batch_size = features.size(0)

            init_features = self.pool(features).squeeze()
            hidden = self.init_hidden(init_features)
            memory = self.init_memory(init_features)

            features = features.permute(0, 2, 3, 1)
            features  = features.view(batch_size, -1, self.encoder_size)

            predictions = torch.Tensor([1]).long().to(self.embed.weight.device).expand(batch_size, 1).long()
            attention_weights = torch.zeros(batch_size, max_seq_length, self.num_pixels).to(self.embed.weight.device)
            
            # initialize list of current beams
            curr_beams = [BeamNode(word_ids = torch.Tensor([1]).long().to(self.embed.weight.device).expand(batch_size).long(),
                                    scores = torch.Tensor([0]).long().to(self.embed.weight.device).expand(batch_size).long(),
                                    seq = [[1] for _ in range(batch_size)])]
            
            for i in range(max_seq_length):
                next_candidates = [[] for _ in range(batch_size)] # stores tuples of (score, word_idx, sequence)
                
                # for each current beam, generate the next `beam_size` best beams.
                # of the `beam_size` * `beam_size` beams generated, only keep the top `beam_size` beams.
                for beam in curr_beams:
                    embeddings = self.embed(beam.word_ids)
                    attention_weighted_encoding, attention_weight = self.attention(features, hidden)
                    gate = self.sigmoid(self.beta(hidden))
                    attention_weighted_encoding = gate * attention_weighted_encoding
                    hidden, memory = self.lstm(
                        torch.cat([embeddings, attention_weighted_encoding], dim=1),
                        (hidden, memory))  # (batch_size_t, decoder_dim)
                    out = self.out(self.dropout(hidden))
                    topv, topi = out.topk(beam_size) # topv = topk scores, topi = topk idxs
                    for k in range(beam_size):
                        for j in range(batch_size):
                            next_candidates[j].append(
                                (beam.scores[j].item() + topv[j][k].item(), 
                                 topi[j][k].item(), 
                                 beam.seq[j] + [topi[j][k].item()])
                            )
                            if len(next_candidates[j]) > beam_size:
                                next_candidates[j].remove(min(next_candidates[j])) # only the top `beam_size` candidates are needed
                curr_beams = [] # reset curr_beams list, create new one from next_candidates
                for k in range(beam_size):
                    word_ids = [next_candidates[j][k][1] for j in range(batch_size)]
                    scores = [next_candidates[j][k][0] for j in range(batch_size)]
                    seq = [next_candidates[j][k][2] for j in range(batch_size)]
                    curr_beams.append(BeamNode(word_ids = torch.LongTensor(word_ids),
                                              scores = torch.FloatTensor(scores),
                                              seq = seq))
                    
                attention_weights[:, i, :] = attention_weight
            # set imgs/targets if provided
            for beam in curr_beams:
                if type(imgs) != type(None):
                    beam.add_imgs(imgs)
                if type(targets) != type(None):
                    beam.add_targets(targets)
            if return_attention:
                return curr_beams, attention_weights
            else:
                return curr_beams
