# TransformerCVAE model 

A further CVAE model for language generation, containing the pretrained language models as encoder and decoder. 
Based on https://github.com/fangleai/TransformerCVAE 

## encoder
bert, roberta, deberta 

## decoder
gpt2, bert

## dataset
ROC dataset for story completion


## how to run
to implement CVAE model with bert-base as encoder and gpt2 as decoder, run `multi_train_base.py --encoder_name=bert-base --decoder_name=gpt2-base` 

