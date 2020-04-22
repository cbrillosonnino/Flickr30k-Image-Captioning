import math

def extract_ngrams(line, max_order, min_order=1):
    ngrams = Counter()
    ngrams.clear()
    for n in range(min_order, max_order + 1):
        for i in range(0, len(line) - n + 1):
            ngram = ' '.join(line[i: i + n])
            ngrams[ngram] += 1
    return ngrams

def ref_stats(output, refs, max_order):
    ngrams = Counter()
    closest_diff = None
    closest_len = None
    for ref in refs:
        reflen = len(ref)
        diff = abs(len(output) - reflen)
        if closest_diff is None or diff < closest_diff:
            closest_diff = diff
            closest_len = reflen
        elif diff == closest_diff:
            if reflen < closest_len:
                closest_len = reflen

        ngrams_ref = extract_ngrams(ref, max_order = max_order)
        for ngram in ngrams_ref.keys():
            ngrams[ngram] = max(ngrams[ngram], ngrams_ref[ngram])

    return ngrams, closest_diff, closest_len

def corpus_bleu(out_stream, ref_streams, ngrams, smooth='exp', smooth_floor=0.01, 
                force=False, lowercase=False, use_effective_order=False):
    """Produces BLEU scores along with its sufficient statistics from a source against one or more references.
    :param sys_stream: The system stream (a sequence of segments). 
    :param ref_streams: A list of one or more reference streams (each a sequence of segments)
    :param smooth: The smoothing method to use
    :param smooth_floor: For 'floor' smoothing, the floor to use
    :param force: Ignore data that looks already tokenized
    :param lowercase: Lowercase the data
    """

    out_len = 0
    ref_len = 0

    correct = [0 for n in range(ngrams)]
    total = [0 for n in range(ngrams)]
    fhs = [out_stream] + ref_streams
    for lines in zip_longest(*fhs):
        if lines[0] is None or lines[1] is None:
            raise EOFError("Source and reference streams have different lengths!")

        output = lines[0]
        refs = lines[1]
        
        out_ngrams = extract_ngrams(output, max_order = ngrams)
        out_len += len(output)

        ref_ngrams, closest_diff, closest_len = ref_stats(output, [refs], max_order = ngrams)
        ref_len += closest_len

        for ngram in out_ngrams.keys():
            n = len(ngram.split())
            correct[n-1] += min(out_ngrams[ngram], ref_ngrams.get(ngram, 0))
            total[n-1] += out_ngrams[ngram]
    return compute_bleu(correct, total, ngrams, out_len, ref_len, smooth, smooth_floor, use_effective_order)

def compute_bleu(correct, total, ngrams, out_len, ref_len, 
                 smooth = 'floor', smooth_floor = 0.01, use_effective_order = False):
    """Computes BLEU score from its sufficient statistics. Adds smoothing.
    :param correct: List of counts of correct ngrams, 1 <= n <= NGRAM_ORDER
    :param total: List of counts of total ngrams, 1 <= n <= NGRAM_ORDER
    :param out_len: The cumulative system length
    :param ref_len: The cumulative reference length
    :param smooth: The smoothing method to use
    :param smooth_floor: The smoothing value added, if smooth method 'floor' is used
    :param use_effective_order: Use effective order.
    """

    precisions = [0 for x in range(ngrams)]

    smooth_mteval = 1.
    effective_order = ngrams
    for n in range(ngrams):
        if total[n] == 0:
            break

        if use_effective_order:
            effective_order = n + 1
        if correct[n] == 0:
            if smooth == 'exp':
                smooth_mteval *= 2
                precisions[n] = 100. / (smooth_mteval * total[n])
            elif smooth == 'floor':
                precisions[n] = 100. * smooth_floor / total[n]
        else:
            precisions[n] = 100. * correct[n] / total[n]
    brevity_penalty = 1.0
    if out_len < ref_len:
        brevity_penalty = math.exp(1 - ref_len / out_len) if out_len > 0 else 0.0

    bleu = brevity_penalty * math.exp(sum(map(lambda x: -9999999999 if x == 0.0 else math.log(x), precisions[:effective_order])) / effective_order)
    return bleu


def bleu_eval(encoder, decoder, data_loader, batch_size):
    with torch.no_grad():
        true_outputs = []
        decoder_outputs = []
        for i, (images, captions, lengths) in enumerate(data_loader):
            if i * batch_size >= 10000 or len(images) != batch_size:
                continue
            true_outputs.extend([[str(tok.item()) for tok in out if tok != 0] for out in captions])

            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            sample = decoder.sample(features, np.shape(captions)[1])
            decoder_outputs.extend(sample.numpy().astype(str).tolist())

        return corpus_bleu(decoder_outputs, [true_outputs], 1)