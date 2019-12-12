import string
import nltk
#nltk.download("all")
from nltk.corpus import treebank
from nltk.tree import Tree
from nltk.tree import ParentedTree
from typing import List, Dict, Tuple, Any
from collections import Counter
import numpy as np
from tqdm import tqdm



def substring(a,b):
    for x in a:
        if x not in b:
            return False
    return True


instances = []  
for file in treebank.fileids()[:150]:
    for tree in treebank.parsed_sents(file):
        instance = {}
        dic = {}
    
        leaves = tree.leaves()
        leaves = [x for x in leaves if substring(x,string.punctuation) == False]
        leaves = [x for x in leaves if not x.startswith('*') and not x=='0']
    
        newtree = ParentedTree.convert(tree)
    
        for subtree in newtree.subtrees(lambda t: t.height() == 2):
            if subtree.leaves()[0] in leaves:
                right = []
                r = subtree.right_sibling()
                while r:
                    for x in r.leaves():
                        if x in leaves:
                            right.append(x)
                    r = r.right_sibling()                                       
      
                left = []        
                l = subtree.left_sibling()
        
                while l:
                    for x in l.leaves():
                        if x in leaves:
                            left.append(x)
                    l = l.left_sibling()             
    
                dic[subtree.leaves()[0]] = left + right
        instance['sentence'] = leaves
        instance['relations'] = dic 
        instances.append(instance)

instances_test = []
for file in treebank.fileids()[150:]:
    for tree in treebank.parsed_sents(file):
        instance = {}
        dic = {}
    
        leaves = tree.leaves()
        leaves = [x for x in leaves if substring(x,string.punctuation) == False]
        leaves = [x for x in leaves if not x.startswith('*') and not x=='0']
    
        newtree = ParentedTree.convert(tree)
    
        for subtree in newtree.subtrees(lambda t: t.height() == 2):
            if subtree.leaves()[0] in leaves:
                right = []
                r = subtree.right_sibling()
                while r:
                    for x in r.leaves():
                        if x in leaves:
                            right.append(x)
                    r = r.right_sibling()                                       
      
                left = []        
                l = subtree.left_sibling()
        
                while l:
                    for x in l.leaves():
                        if x in leaves:
                            left.append(x)
                    l = l.left_sibling()             
    
                dic[subtree.leaves()[0]] = left + right
        instance['sentence'] = leaves
        instance['relations'] = dic 
        instances_test.append(instance)
        
def build_vocabulary(instances: List[Dict],
                     vocab_size: 10000, add_tokens: List[str] = None):
                    # add_tokens: List[str] = None) -> Tuple[Dict, Dict]:
    """
    Parameters
    ----------
    instances : ``List[Dict]``
        List of instance returned by read_instances from which we want
        to build the vocabulary.
    vocab_size : ``int``
        Maximum size of vocabulary
    add_tokens : ``List[str]``
        if passed, those words will be added to vocabulary first.
    """
    print("\nBuilding Vocabulary.")

    # make sure pad_token is on index 0
    UNK_TOKEN = "@UNK@"
    PAD_TOKEN = "@PAD@"
    token_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1}

    # First add tokens which were explicitly passed.
    add_tokens = add_tokens or []
    for token in add_tokens:
        if not token.lower() in token_to_id:
            token_to_id[token] = len(token_to_id)

    # Add remaining tokens from the instances as the space permits
    words = []
    for instance in instances:
        words.extend(instance['sentence'])
    token_counts = dict(Counter(words).most_common(vocab_size))
    for token, _ in token_counts.items():
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)
        if len(token_to_id) == vocab_size:
            break
    # Make reverse vocabulary lookup
    id_to_token = dict(zip(token_to_id.values(), token_to_id.keys()))
    return (token_to_id, id_to_token)


def save_vocabulary(vocab_id_to_token: Dict[int, str], vocabulary_path: str) -> None:
    """
    Saves vocabulary to vocabulary_path.
    """
    with open(vocabulary_path, "w") as file:
        # line number is the index of the token
        for idx in range(len(vocab_id_to_token)):
            file.write(vocab_id_to_token[idx] + "\n")

def load_vocabulary(vocabulary_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Loads vocabulary from vocabulary_path.
    """
    vocab_id_to_token = {}
    vocab_token_to_id = {}
    with open(vocabulary_path, "r") as file:
        for index, token in enumerate(file):
            token = token.strip()
            if not token:
                continue
            vocab_id_to_token[index] = token
            vocab_token_to_id[token] = index
    return (vocab_token_to_id, vocab_id_to_token)

def load_glove_embeddings(embeddings_txt_file: str,
                          embedding_dim: int,
                          vocab_id_to_token: Dict[int, str]) -> np.ndarray:
    """
    Given a vocabulary (mapping from index to token), this function builds
    an embedding matrix of vocabulary size in which ith row vector is an
    entry from pretrained embeddings (loaded from embeddings_txt_file).
    """
    tokens_to_keep = set(vocab_id_to_token.values())
    vocab_size = len(vocab_id_to_token)

    embeddings = {}
    print("\nReading pretrained embedding file.")
    with open(embeddings_txt_file) as file:
        for line in tqdm(file):
            line = str(line).strip()
            token = line.split(' ', 1)[0]
            if not token in tokens_to_keep:
                continue
            fields = line.rstrip().split(' ')
            if len(fields) - 1 != embedding_dim:
                raise Exception(f"Pretrained embedding vector and expected "
                                f"embedding_dim do not match for {token}.")
                continue
            vector = np.asarray(fields[1:], dtype='float32')
            embeddings[token] = vector

    # Estimate mean and std variation in embeddings and initialize it random normally with it
    all_embeddings = np.asarray(list(embeddings.values()))
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_std = float(np.std(all_embeddings))

    embedding_matrix = np.random.normal(embeddings_mean, embeddings_std,
                                        (vocab_size, embedding_dim))
    embedding_matrix = np.asarray(embedding_matrix, dtype='float32')

    for idx, token in vocab_id_to_token.items():
        if token in embeddings:
            embedding_matrix[idx] = embeddings[token]

    return embedding_matrix
#import pdb;pdb.set_trace()
def index_instances(instances: List[Dict], token_to_id: Dict) -> List[Dict]:
    """
    Uses the vocabulary to index the fields of the instances. This function
    prepares the instances to be tensorized.
    """
    for instance in instances:
        sentence_ids = []
        for token in instance['sentence']:
            if token in token_to_id:
                sentence_ids.append(token_to_id[token])
            else:
                sentence_ids.append(0) # 0 is index for UNK
        instance['sentence_ids'] = sentence_ids
        instance.pop("sentence")
        keys = list(instance['relations'].keys())
        values = list(instance['relations'].values())
        key_ids = []
        for token in keys:
            if token in token_to_id:
                key_ids.append(token_to_id[token])   
            else:
                key_ids.append(0)    
        value_ids = []
        for value in values:
            value_id = []
            for token in value:
                if token in token_to_id:
                    value_id.append(token_to_id[token])   
                else:
                    value_id.append(0)
            value_ids.append(value_id)
        instance['relations_ids'] = dict(zip(key_ids,value_ids))
        instance.pop('relations')
    return instances


def generate_batches(instances: List[Dict], batch_size) -> List[Dict[str, np.ndarray]]:
    """
    Generates and returns batch of tensorized instances in a chunk of batch_size.
    """
    def chunk(items: List[Any], num: int):
        return [items[index:index+num] for index in range(0, len(items), num)]
    batches_of_instances = chunk(instances, batch_size) # List[List[Dict]]

    batches = []
    for batch_of_instances in tqdm(batches_of_instances): # List[Dict]

        num_token_ids = [len(instance['sentence_ids'])
                         for instance in batch_of_instances]
        #max_num_token_ids = 200
        max_num_token_ids = max(num_token_ids)

        count = min(batch_size, len(batch_of_instances))
        batch = {"inputs": np.zeros((count, max_num_token_ids), dtype=np.int32)}
        batch["labels"]= np.zeros((count, max_num_token_ids, max_num_token_ids), dtype=np.int32)
        
        '''
        if "labels" in  batch_of_instances[0]:
            batch["labels"] = np.zeros(count, dtype=np.int32)
        '''
        for batch_index, instance in enumerate(batch_of_instances):
            num_tokens = len(instance['sentence_ids'])
            inputs = np.array(instance['sentence_ids'])
            batch["inputs"][batch_index][:num_tokens] = inputs
            labels = np.zeros((max_num_token_ids,max_num_token_ids),dtype=np.int32)
            for word_index, word_id in enumerate(instance['sentence_ids']):
                c_list = instance['relations_ids'][word_id]     
                
                for label_index, word_id in enumerate(instance['sentence_ids']):
                    if word_id in c_list:
                        labels[word_index][label_index] = 1
            batch["labels"][batch_index] = labels
            
                
            #if "labels" in instance:
                #batch["labels"][batch_index] = np.array(instance["labels"])
        batches.append(batch)

    return batches



