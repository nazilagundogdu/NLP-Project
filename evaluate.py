#pylint: disable = invalid-name
# inbuilt lib imports:
import json
import argparse
import os
from Untitled import instances_test, generate_batches,index_instances,build_vocabulary,load_vocabulary
from collections import Counter
def evaluate(prediction_data_path: str) -> float:
    """
        Evaluates accuracy of label predictions in ``prediction_data_path``
        based on gold labels in ``gold_data_path``.
    """
    '''
        with open(gold_data_path) as file:
            gold_labels = [int(json.loads(line.strip())["label"])
                           for line in file.readlines() if line.strip()]
    '''
    VOCAB_SIZE = 10000
    '''
    vocab_path = os.path.join(args.load_serialization_dir, "vocab.txt")
    vocab_token_to_id, vocab_id_to_token = load_vocabulary(vocab_path)
    '''
    GLOVE_COMMON_WORDS_PATH = os.path.join("data", "glove_common_words.txt")
    with open(GLOVE_COMMON_WORDS_PATH) as file:
        glove_common_words = [line.strip() for line in file.readlines() if line.strip()]
    vocab_token_to_id, vocab_id_to_token = build_vocabulary(instances_test, VOCAB_SIZE,
                                                                glove_common_words)
    
    gold_labels_matrix = generate_batches(index_instances(instances_test, vocab_token_to_id),len(instances_test))[0]["labels"]
    import pdb;pdb.set_trace()
    labels_list = []
    for array in gold_labels_matrix:
        for sequence in array:
            labels_list.append([x for x in sequence])
            
    file = open(prediction_data_path)
    ''' 
    predicted_labels = [int(line.strip())
                            for line in file.readlines() if line.strip()]
    '''
    #import pdb;pdb.set_trace()
    predicted = []
    count = []
    for index, line in enumerate(file.readlines()):
        line = line.strip()
        predicted_labels = line.split()
        #import pdb;pdb.set_trace()
        for i in range(len(predicted_labels)):
            if int(predicted_labels[i]) == labels_list[index][i]:
                count.append(1.0)
            else:
                count.append(0.0)
        predicted += predicted_labels
            
    '''        
    correct_count = sum([1.0 if predicted_label == gold_label else 0.0
                         for predicted_label, gold_label in zip(predicted_labels, gold_labels)])
    '''
    correct_count = sum(count)
    total_count = len(predicted)
    print(correct_count / total_count)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate classification predictions.')
    parser.add_argument('gold_data_path', type=str, help='gold data file path.')
    parser.add_argument('prediction_data_path', type=str,
                        help='predictions data file path.')

    args = parser.parse_args()
    accuracy = evaluate(args.prediction_data_path)
    
    #print(f"Accuracy: {round(accuracy, 2)}")
