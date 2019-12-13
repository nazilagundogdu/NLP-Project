# NLP-Project
# A c-command decoder

This project is a c-command classifier over strings. The data is extracted from Penn Treebank. The list of c-commanding words were exracted from the parsed trees in the PTB, using nltk tools. Giving the sentence and this list (stored as a dictionary) to the model, the model classifies words into two categories of 1 and 0, with 1 meaning the two words are in c-command relation and 0 meaning they are not. 

data.py: reads the data from Penn Treebank and generates instances and labels

sequence_to_vector.py: implements the GRU model and the Dense layer, gives back sentence representation and our stacked output

main.py: is the main classifier

train.py: trains the model

loss.py: computes the model's loss using cross entropy loss

predict.py: does the prediction

evaluate.py: does the evaluation

# To implement the model, use the following steps:

After downloading all the files in the repo:

1. You need to have conda, which can be installed at:
https://conda.io/projects/conda/en/latest/user-guide/install/index.html

2. Create the environment on your terminal:
conda create -n nlp-project python=3.6

3. Activate the environment:
conda activate nlp-project

4. Install the requirements:
pip install -r requirements.txt

5. Download glove word vectors:
./download_glove.sh

6. Train the model:

python train.py main \
                  --seq2vec-choice gru \
                  --embedding-dim 50 \
                  --num-layers 4 \
                  --num-epochs 5 \
                  --suffix-name _gru_5k_with_emb \
                  --pretrained-embedding-file glove.6B.50d.txt
                  
The parameters that can be changed: 
    num-epochs
    num-layers
    embedding-dim can be set to: 100, 200, 300 --> if this is changed, the pretrained-embedding-file should match the used dimension
    
7. Predict:
python predict.py serialization_dirs/main_gru_5k_with_emb predictions-file my_predictions.txt

8. Evaluate:
python evaluate.py my_predictions.txt
    
