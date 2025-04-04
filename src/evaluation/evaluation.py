# use meteor metric to evaluate the generated outputs
import nltk
nltk.download('punkt')
from nltk.tokenize import RegexpTokenizer
import evaluate
import numpy as np

def evaluate_outputs(predictions, references):
    meteor = evaluate.load('meteor')
    text_preds = [(p if str(p).endswith(("!", "！", "?", "？", "。")) else str(p) + "。") for p in predictions]
    text_labels = [(l if l.endswith(("!", "！", "?", "？", "。")) else l + "。") for l in references]
    sent_tokenizer_jp = RegexpTokenizer(u'[^!！?？。]*[!！?？。]')
    
    text_preds = ["\n".join(np.char.strip(sent_tokenizer_jp.tokenize(p))) for p in text_preds]
    text_labels = ["\n".join(np.char.strip(sent_tokenizer_jp.tokenize(l))) for l in text_labels]
    
    return meteor.compute(predictions=text_preds, references=text_labels)