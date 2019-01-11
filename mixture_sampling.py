import torch
import pickle
from generate import evaluate

def beam_generation(pairs, encoder, decoder, voc, args):
    generated_pairs = []
    counter = 0
    for i in range(len(pairs)):

       temp = [pairs[i][0]]
       ai_response = evaluate(encoder, decoder, voc, sentence=pairs[i][0], beam=args.beam_size)
       temp.append(ai_response)

       generated_pairs.append(temp)
       if counter % 1000 == 0:
           print('Now generated {} sentence pairs, remaining {} sentence pairs'.format
                 (str(counter), str(len(pairs) - counter)))

    return generated_pairs
