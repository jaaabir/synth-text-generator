import os 
import numpy as np 

def reformat_txt_file(input_file, output_file, max_words=6):
    """
    Reads a text file line by line, splits lines longer than `max_words` words 
    into chunks of size <= max_words, and writes reformatted lines into a new file.
    
    :param input_file: str, path to input .txt file
    :param output_file: str, path to output .txt file
    :param max_words: int, maximum number of words per line (default=6)
    """
    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:
        
        for line in infile:
            words = line.strip().split()
            mw = np.random.randint(1, max_words)
            # Split into chunks of up to max_words
            for i in range(0, len(words), mw):
                chunk = words[i:i + mw]
                outfile.write(" ".join(chunk) + "\n")


ip_fname = input('Input file name: ')
ip_fname = os.path.join('.', 'resources', 'corpus', ip_fname)
op_fname = ip_fname.replace('.txt','') + '_oneliner.txt'

reformat_txt_file(ip_fname, op_fname)
