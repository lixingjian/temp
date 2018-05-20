import sys, os, random

def read_all(dirname):
    labels = []
    data = []
    for label in os.listdir(dirname):
        labels.append(label)
        class_dir = dirname + '/' + label
        for image in os.listdir(class_dir):
            image_path = class_dir + '/' + image
            data.append((image_path, label))
    return data, labels

indir = sys.argv[1]
outdir = sys.argv[2]
train_rate = float(sys.argv[3])
valid_rate = float(sys.argv[4])

data, labels = read_all(indir)
random.shuffle(data)
print(len(data), len(labels))

def osrun(cmd, debug = 1):
    if debug:
        print(cmd)
    os.system(cmd)

def write_data(dirname, pos_begin, pos_end):
    cmd = 'mkdir -p %s/%s' % (outdir, dirname)
    osrun(cmd)
    dirname = outdir + '/' + dirname
    for label in labels:
        os.mkdir(dirname + '/' + label)
    n = len(data)
    n1 = int(pos_begin * n)
    n2 = int(pos_end * n)
    for i in range(n1, n2):
        image, label = data[i]
        cmd = 'cp %s %s/%s' % (image, dirname, label)
        osrun(cmd)

write_data('train', 0, train_rate)
write_data('valid', train_rate, train_rate + valid_rate)
write_data('test', train_rate + valid_rate, 1)
 
