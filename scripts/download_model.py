import subprocess
import os
from transformers import AutoModelForSequenceClassification
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--bart_large_only', action='store_true')
args = parser.parse_args()

print('Downloading bart_large ...')
bart_large = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
bart_large.save_pretrained('models/bart_large.h5')
if args.bart_large_only is False:
    print('Downloading bert ...')
    os.makedirs("tmp", exist_ok=True)
    subprocess.run(['wget', 'https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip', '-O',
                    'tmp/uncased_L-12_H-768_A-12.zip'])
    subprocess.run(['unzip', 'tmp/uncased_L-12_H-768_A-12.zip', '-d', 'tmp/uncased_L-12_H-768_A-12'])
    os.makedirs("tmp/model", exist_ok=True)
    subprocess.run(['mv', 'tmp/uncased_L-12_H-768_A-12', 'tmp/model'])



