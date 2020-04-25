'''
    Compute RGB mean and standard deviation given set of images
    Usage: ImageMeanStdev.py filepath method

    Inputs:
        filepath, e.g. 'flickr30k-images', which is folder containing images
            with which to calculate RGB mean and standard deviation
        method, {'serial', 'threading', 'multiprocessing'}. default 'threading'

    Output;
        [[[r_mean, b_mean, g_mean], [r_stdev, b_stdev, g_stdev]],
         [[r_mean/255, b_mean/255, g_mean/255], [r_stdev/255, b_stdev/255, g_stdev/255]]]
'''

import os
import numpy as np
from PIL import Image

def meanStdevCalcs(filepath, method):
    if method = 'serial':
        #mean
        r_channel_sum = 0
        g_channel_sum = 0
        b_channel_sum = 0
        count = 0
        for filename in os.listdir(filepath):
            if filename[-3:] == 'jpg':
                img = np.array(Image.open(os.path.join(filepath, filename)).convert('RGB'))
                r_channel_sum += np.sum(img[:,:,0])
                g_channel_sum += np.sum(img[:,:,1])
                b_channel_sum += np.sum(img[:,:,2])
                count += img.shape[0] * img.shape[1]
        r_mean = r_channel_sum/count
        g_mean = g_channel_sum/count
        b_mean = b_channel_sum/count

        #stdev
        r_channel_sum = 0
        g_channel_sum = 0
        b_channel_sum = 0
        count = 0
        for filename in os.listdir(filepath):
            if filename[-3:] == 'jpg':
                img = np.array(Image.open(os.path.join(filepath, filename)).convert('RGB'))
                r_channel_sum += np.sum(np.square(img[:,:,0] - r_mean))
                g_channel_sum += np.sum(np.square(img[:,:,1] - g_mean))
                b_channel_sum += np.sum(np.square(img[:,:,2] - b_mean))
                count += img.shape[0] * img.shape[1]
        r_stdev = np.sqrt(r_channel_sum/count)
        g_stdev = np.sqrt(g_channel_sum/count)
        b_stdev = np.sqrt(b_channel_sum/count)

    elif method = 'multiprocessing':
        #mean
        img_files = [filename for filename in os.listdir(filepath)]
        chunks = [(img_files[i:i+500]) for i in range(0, len(img_files), 500)]
        def getRGB(chunk):
            r_channel_sum = 0
            g_channel_sum = 0
            b_channel_sum = 0
            count = 0
            for filename in chunk:
                if filename[-3:] == 'jpg':
                    img = np.array(Image.open(os.path.join(filepath, filename)).convert('RGB'))
                    r_channel_sum += np.sum(img[:,:,0])
                    g_channel_sum += np.sum(img[:,:,1])
                    b_channel_sum += np.sum(img[:,:,2])
                    count += img.shape[0] * img.shape[1]
            return (r_channel_sum, g_channel_sum, b_channel_sum, count)
        from multiprocessing.pool import Pool
        with Pool(4) as p:
            res = p.map(getRGB, chunks)
        results = np.array(res).sum(axis=0)
        r,g,b,c = results[0], results[1], results[2], results[3]
        r_mean = r/c
        g_mean = g/c
        b_mean = b/c

        #stdev
        def getRGB(chunk):
            r_channel_sum = 0
            g_channel_sum = 0
            b_channel_sum = 0
            count = 0
            for filename in chunk:
                if filename[-3:] == 'jpg':
                    img = np.array(Image.open(os.path.join(filepath, filename)).convert('RGB'))
                    r_channel_sum += np.sum(np.square(img[:,:,0] - r_mean))
                    g_channel_sum += np.sum(np.square(img[:,:,1] - g_mean))
                    b_channel_sum += np.sum(np.square(img[:,:,2] - b_mean))
                    count += img.shape[0] * img.shape[1]
            return (r_channel_sum, g_channel_sum, b_channel_sum, count)
        with Pool(4) as p:
            res = p.map(getRGB, chunks)
        results = np.array(res).sum(axis=0)
        r,g,b,c = results[0], results[1], results[2], results[3]
        r_stdev = np.sqrt(r/c)
        g_stdev = np.sqrt(g/c)
        b_stdev = np.sqrt(b/c)

    else: #else: threading
        #mean
        from queue import Queue
        from threading import Thread
        from threading import Lock
        img_files = [filename for filename in os.listdir(filepath)]
        chunks = [(img_files[i:i+500]) for i in range(0, len(img_files), 500)]
        res = []
        def thread_worker(q):
            while True:
                filename = q.get()
                if filename[-3:] == 'jpg':
                    img = np.array(Image.open(os.path.join(filepath, filename)).convert('RGB'))
                    r = np.sum(img[:,:,0])
                    g = np.sum(img[:,:,1])
                    b = np.sum(img[:,:,2])
                    count = img.shape[0] * img.shape[1]
                    res.append([r,g,b,count])
                q.task_done()

        q = Queue()
        num_threads = 4
        start_time = time.time()
        for i in range(num_threads):
            worker = Thread(target=thread_worker, args=(q, ))
            worker.setDaemon(True)
            worker.start()
        for filename in img_files:
            q.put(filename)
        q.join()
        results = np.array(res).sum(axis=0)
        r,g,b,c = results[0], results[1], results[2], results[3]
        r_mean = r/c
        g_mean = g/c
        b_mean = b/c

        #stdev
        res = []
        def thread_worker(q):
            while True:
                filename = q.get()
                if filename[-3:] == 'jpg':
                    img = np.array(Image.open(os.path.join(filepath, filename)).convert('RGB'))
                    r = np.sum(np.square(img[:,:,0] - r_mean))
                    g = np.sum(np.square(img[:,:,1] - g_mean))
                    b = np.sum(np.square(img[:,:,2] - b_mean))
                    count = img.shape[0] * img.shape[1]
                    res.append([r,g,b,count])
                q.task_done()

        q = Queue()
        num_threads = 4
        start_time = time.time()
        for i in range(num_threads):
            worker = Thread(target=thread_worker, args=(q, ))
            worker.setDaemon(True)
            worker.start()
        for filename in img_files:
            q.put(filename)
        q.join()
        results = np.array(res).sum(axis=0)
        r,g,b,c = results[0], results[1], results[2], results[3]
        r_stdev = np.sqrt(r/c)
        g_stdev = np.sqrt(g/c)
        b_stdev = np.sqrt(b/c)

    output = [[[r_mean, b_mean, g_mean], [r_stdev, b_stdev, g_stdev]],
             [[r_mean/255, b_mean/255, g_mean/255], [r_stdev/255, b_stdev/255, g_stdev/255]]]
    return output

# Enter this block if we're in main
if __name__ == "__main__":
    filepath = sys.argv[1]
    try:
        method = sys.argv[2]
    except:
        method = 'threading'
    res = meanStdevCalcs(filepath, method)
