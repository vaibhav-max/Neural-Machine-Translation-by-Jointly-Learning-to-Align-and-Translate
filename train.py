from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, random_split
from dataset import BilingualDataset
from code import getModel
import torch
import torch.nn as nn
import warnings
from tqdm import tqdm
from config import get_config, get_weights_file_path
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
import os


def greedy_decode(model, source, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output, encoder_hidden = model.encode(source)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # calculate output
        out = model.decode(decoder_input, config["seq_len"], encoder_hidden, encoder_output)
        print(out[:,-1,:].shape)
        # get next token
        # prob = model.linear_layer(out[:, -1])
        _, next_word = torch.max(out[:,-1,:], dim=-1)
        
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )
        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            #print_msg('{}'.format(len(batch)))

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")
            cmpr = []
            cmpr.append(target_text.split(' '))
            print_msg('{} {}'.format(cmpr, model_out_text.split(' ')))
            #print_msg('BLEU score -> {}'.format(round(sentence_bleu(cmpr, model_out_text.split(' ')), 2)))
            #BLEUscore = nltk.translate.bleu_score.sentence_bleu([target_text], model_out_text)
            #print(BLEUscore)

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.text.CharErrorRate()
        cer = metric(predicted, expected)
        #writer.add_scalar('validation cer', cer, global_step)
        #writer.flush()

        # Compute the word error rate
        metric = torchmetrics.text.WordErrorRate()
        wer = metric(predicted, expected)
        #writer.add_scalar('validation wer', wer, global_step)
        #writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.text.BLEUScore()
        bleu = metric(predicted, expected)
        #writer.add_scalar('validation BLEU', bleu, global_step)
        #writer.flush()


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]
        
def get_or_build_tokenizers(config, ds, lang) :
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else :
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset("wmt14","hi-en", split='train')
    
    tokenizer_src = get_or_build_tokenizers(config= config, ds=ds_raw,lang= "en")
    tokenizer_tgt = get_or_build_tokenizers(config= config,ds= ds_raw,lang= "hi")
    
    train_ds_size = int(0.9* len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # print("Here")
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, "en", "hi", config["seq_len"])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, "en", "hi", config["seq_len"])
    
    max_len_src = 0
    max_len_tgt = 0
    
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation']["en"]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"]["hi"]).ids
        
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
        
    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")
    
    train_dataloader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, src_vocab_size, tgt_vocab_size):
    model = getModel(config['seq_len'], config['seq_len'],src_vocab_size, tgt_vocab_size, config['d_model'], config['encoder_hidden_size'], config['decoder_hidden_size'], config['d_ff'], config['l'])
    return model
    # src_seq_length, tgt_seq_length, src_vocab_size, tgt_vocab_size, d_model, encoder_hidden_size, decoder_hidden_size, d_ff, l
    
def train_model(config) :
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Make sure the weights folder exists
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            
            encoder_output, encoder_hidden = model.encode(encoder_input)
            decoder_output = model.decode(decoder_input, config["seq_len"], encoder_hidden, encoder_output)
            
            
            label = batch['label'].to(device)
            loss = loss_fn(decoder_output.to(device).view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item(): 6.3f}"})
            
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()
            
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad(set_to_none= True)
            
            global_step+=1
        
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
          
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
            

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)