# use meteor metric to evaluate the generated outputs
from evaluate import evaluate

def evaluate_outputs(predictions, references):
    meteor = evaluate.load('meteor')
    return meteor.compute(predictions=predictions, references=references)