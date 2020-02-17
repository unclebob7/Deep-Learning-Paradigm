import math
import sys
import os
import glob
import json

def save_hparams(hparams, path):
    '''
    Saves hparams to path
    '''
    if not os.path.exists(path): os.makedirs(path)
    hp = json.dumps(vars(hparams))
    with open(os.path.join(path, "hparams"), 'w', encoding='utf-8') as fout:
        fout.write(hp)

# def timer(start,end):
#    hours, rem = divmod(end-start, 3600)
#    minutes, seconds = divmod(rem, 60)
#    return "{:0>2}:{:0>2}:{:02d}".format(int(hours),int(minutes),int(seconds))

def view_bar(message, num, total, accur, elapse):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    hours, rem = divmod(elapse, 3600)
    minutes, seconds = divmod(rem, 60)
    elapse_format = "\t{:0>2}:{:0>2}:{:02d}".format(int(hours),int(minutes),int(seconds))
    r = '\r%s:[%s%s]%d%%\t%d/%d\t%.2f%%' % (
    message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total, accur * 100)
    r = r + elapse_format
    sys.stdout.write(r)
    sys.stdout.flush()

def latest_ckpt(dir):
    if not os.path.exists(dir): os.makedirs(dir)
    ckpt_ls = glob.glob("./%s/*.pth" % (dir))   # the serialized ckpt file end with ".pt"
    if ckpt_ls == []:
        return None
    else:
        last_ckpt = max(ckpt_ls, key=os.path.getctime)
        return last_ckpt


