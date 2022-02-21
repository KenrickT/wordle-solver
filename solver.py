# %%
# wordlist from https://github.com/dwyl/english-words/blob/master/words_alpha.txt
from alive_progress import alive_bar
from wordfreq import zipf_frequency
from collections import Counter
import pandas as pd
import emoji
import spacy
nlp = spacy.load("en_core_web_sm")

# %%
# make sure that txt file is in the same folder as py file
df = pd.read_csv('words_alpha.txt', sep=" ", header=None)
df.columns = ['word']

# get only 5-letter-words
df_fives = df.loc[df['word'].str.len() == 5].reset_index(drop=True)
words5L = df_fives['word'].tolist()

# words without repeating letters
words5Lnorepeat = [x for x in words5L if len(set(x))==len(x)]

# %%
def parse_inputs(word, result):
    '''
    Take the input words (e.g. "DRAFT") and results (e.g. "BBYBB")
    and split them into record the letters and corresponding positions

    Notes:
        Hard-coded to only iterate through the first 5 inputs
        Does not check if word string length == result string length
        Does not fix / check for upper vs lower case consistency
    '''

    green_letters = []
    green_positions = []
    yellow_letters = []
    yellow_positions = []
    black_letters = []
    black_positions = []

    for i in range(0, 5):
        if result[i] == 'g':
            green_letters.append(word[i])
            green_positions.append(i)
        elif result[i] == 'y':
            yellow_letters.append(word[i])
            yellow_positions.append(i)
        else: 
            black_letters.append(word[i])
            black_positions.append(i)

    return green_letters, green_positions, yellow_letters, yellow_positions, black_letters, black_positions

# %%
# function to check for green words
def get_green_shortlist(wordlist, green_letters, green_positions):

    # check if there are any green letters
    # if none, return original wordlist
    if len(green_letters) > 0:
        shortlist_green = []
        for word in wordlist:    
            matches = 0
            max_match = len(green_letters)

            # look for words that have green letters in the same positions
            for i in range(len(green_letters)):
                position = green_positions[i]
                if word[position] == green_letters[i]:
                    matches += 1
                else: pass
            
            # only include words that have ALL the green letters + positions
            if matches == max_match:
                shortlist_green.append(word)
            else: pass

    else: shortlist_green = wordlist.copy()        
    return shortlist_green

# %%
# checking for yellow words
def get_yellow_shortlist(shortlist_green, yellow_letters, yellow_positions, green_letters, black_positions):

    # check if there are any yellow letters
    # if none, return values from green shortlist
    if len(yellow_letters) > 0:

        shortlist_yellow = []
        for word in shortlist_green:
            matches = 0
            max_match = len(yellow_letters)

            for i in range(len(yellow_letters)):
                position = yellow_positions[i]

                # scenario - one letter is green, one letter is yellow
                if yellow_letters[i] in green_letters:
                    # include word if yellow letter is in black position
                    for posb in black_positions:
                        if yellow_letters[i] == word[posb]:
                            matches += 1

                    # pass if yellow letter is in current yellow position
                    # but include word if yellow letter is in another yellow letter's position
                    for posy in yellow_positions:
                        # do not include if word has the yellow letter position as current yellow letter
                        if i == posy: 
                            pass
                        # include if word has the yellow letter in another yellow letter's position
                        elif yellow_letters[i] == word[posy]:
                            matches += 1
                
                # scenario - yellow letter is present in green letters list
                # include words that contain ALL the yellow letters in general AND not in current yellow letter's position
                elif yellow_letters[i] in word and word[position] != yellow_letters[i]:
                    matches += 1
                else: pass
            
            # only include words that have ALL the yellow letters (not in their own positions)
            if matches == max_match:
                shortlist_yellow.append(word)
            else: pass

    else: shortlist_yellow = shortlist_green.copy()
    return shortlist_yellow

# %%
# checking for black words
def get_black_shortlist(shortlist_yellow, black_letters, black_positions, green_letters, yellow_letters):

    # check if there are any black letters
    # if none, return values from yellow shortlist
    if len(black_letters) > 0:

        shortlist_black = []
        for word in shortlist_yellow:    
            matches = 0
            max_match = len(black_letters)

            for i in range(len(black_letters)):    
                
                # scenario - black letter is also found in green_letters, but not in yellow_letters
                # check if black letter is not in the green letter's position
                if black_letters[i] in green_letters and black_letters[i] not in yellow_letters:
                
                    # remove instances of green letters from word
                    word_without_green = word[:]
                    for j in green_letters:    
                        word_without_green = word_without_green.replace(j, "", 1)
                    # then look for words where black letter is not anymore in word
                    # include only the words without ALL of the black letters
                    if black_letters[i] not in word_without_green:
                        matches += 1
                    else: pass
                
                # scenario - black letter is also found in yellow_letters
                # if black letter is also a yellow letter
                # you can only take out words with black letter in its own position
                elif black_letters[i] in yellow_letters:
                    position = black_positions[i]
                    if word[position] != black_letters[i]:
                        matches += 1
                    else: pass

                # if black letter is not found on green letters nor yellow letters
                # includes words that do not have the black letter
                elif black_letters[i] not in word:
                    matches += 1
                else: pass
            
            if matches == max_match:
                shortlist_black.append(word)
            else: pass

    else: shortlist_black = shortlist_yellow.copy()
    return shortlist_black

# %%
# SCORING
# create a dictionary to map each letter to score based on LETTER FREQUENCY

def create_letter_score_ref(shortlist):
    # combine all wordings into 1 string
    shortword = ''
    for i in shortlist: 
        shortword += i
    # create dict of frequency count per letter
    count_ref = Counter(shortword)
    return count_ref


def create_letter_score_ref_allwords(wordlist):
    # combine all wordings into 1 string
    shortword = ''
    for i in wordlist: 
        shortword += i
    # create dict of frequency count per letter
    count_ref = Counter(shortword)
    return count_ref

#%%
# SCORING
# create a dictionary to map each word to score based on WORD FREQUENCY

def create_wordfreq_score_ref(wordlist):
    wordfreq_ref = {}
    for word in wordlist:
        wordfreq_ref[word] = zipf_frequency(word, 'en')
    return wordfreq_ref

#%%
# SCORING
# create a dictionary to map each word to check if plural vs singular using spacy
# idea is to penalize plural word options (since main wordle so far avoids plural words)

def identify_nonplural_words(wordlist):
    
    nonplural_words = {}
    docs = nlp.pipe(wordlist)
    
    for doc in docs:
        for word in doc:
            tag = word.tag_
            if tag == 'NNS':
                nonplural_words[str(word)] = 0.00
            else: 
                nonplural_words[str(word)] = 0.50
    
    return nonplural_words

# %%
# SCORING
# sort the shortlist by combining scores weighted by importance
def sort_word_score(shortlist, letter_score_local_ref, letter_score_global_ref, word_score_global_ref, word_score_nonplural_ref):

    words = []
    scores_letter_local = []
    scores_letter_global = []
    scores_wordfreq = []
    scores_nonplural = []
    scores_norepeating = []

    for word in shortlist:
        words.append(word)
        
        # get letter score
        letter_score = 0
        letter_score_allwords = 0
        for letter in word:
            letter_score += letter_score_local_ref[letter]
            letter_score_allwords += letter_score_global_ref[letter]
        scores_letter_local.append(letter_score)
        scores_letter_global.append(letter_score_allwords)

        # get wordfreq score
        try:
            wordfreq_score = word_score_global_ref.get(word)
            scores_wordfreq.append(wordfreq_score)
        except:
            scores_wordfreq.append(0)

        # get nonplural score
        nonplural_score = word_score_nonplural_ref.get(word, 0)
        scores_nonplural.append(nonplural_score)

        # add points to words without repeating letters
        if word in words5Lnorepeat:
            scores_norepeating.append(0.50)
        else:
            scores_norepeating.append(0.00)

    # create a dataframe and sort values
    df = pd.DataFrame(zip(words, scores_letter_local, scores_letter_global, scores_wordfreq, scores_nonplural, scores_norepeating))
    df.columns = ['word', 'scores_letter_local', 'scores_letter_global', 'scores_wordfreq', 'scores_nonplural', 'scores_norepeating']

    # create final score
    df['letters_local_weight'] = df['scores_letter_local'].fillna(0).sum()
    df['letters_global_weight'] = df['scores_letter_global'].fillna(0).sum()
    df['wordfreq_weight'] = df['scores_wordfreq'].fillna(0).sum()
    
    # convert to percentage of total
    df['letters_local_weight'] = df['scores_letter_local'] / df['letters_local_weight']
    df['letters_global_weight'] = df['scores_letter_global'] / df['letters_global_weight']
    df['wordfreq_weight'] = df['scores_wordfreq'] / df['wordfreq_weight']
    
    # add weights and sort
    df['final_score'] = df['letters_global_weight']*3 + df['wordfreq_weight']*2 + df['scores_nonplural']*1 + df['scores_norepeating']*1 + df['letters_local_weight']*0.50
    df = df.sort_values(by='final_score', ascending=False).reset_index(drop=True)

    # create sorted shortlist ready for printing
    sorted_shortlist = df['word']
    return sorted_shortlist

# %%
def create_shortlist(word, result, word_reference):

    try:
        green_letters, green_positions, yellow_letters, yellow_positions, black_letters, black_positions = parse_inputs(word, result)
        shortlist_green = get_green_shortlist(word_reference, green_letters, green_positions)
        shortlist_yellow = get_yellow_shortlist(shortlist_green, yellow_letters, yellow_positions, green_letters, black_positions)
        shortlist_black = get_black_shortlist(shortlist_yellow, black_letters, black_positions, green_letters, yellow_letters)
        shortlist_final = list(set(shortlist_black))
        
        letter_score_local_ref = create_letter_score_ref(shortlist_final)
        shortlist_sorted = sort_word_score(shortlist_final, letter_score_local_ref, letter_score_global_ref, word_score_global_ref, word_score_nonplural_ref)
    
    except: shortlist_sorted = []
    return shortlist_sorted
    

# %%
def print_box_colors(word, result):

    green_square = emoji.emojize(':green_square:')
    yellow_square = emoji.emojize(':yellow_square:')
    black_square = emoji.emojize(':black_large_square:')
    white_square = emoji.emojize(':white_large_square:')
    
    # assign square color based on inputs
    box_print = ''
    for res in result:
        if res == 'g':
            box_print += green_square
        elif res == 'y':
            box_print += yellow_square
        elif res =='b': 
            box_print += black_square
        else: box_print += white_square
    
    # convert word to upper case and put space in between each letter
    word_print = ''
    for word in word.upper():
        word_print += word + ' '

    return box_print, word_print

# %%
# Introduction Message
print()
print()
print('********** Hi, welcome to WORDLE SOLVER! **********')
print('> input RESULT in format "gybbb"')
print('> ref g=green, y=yellow, b=black')
print('> other available commands: summary, restart, exit')
print()
print()

with alive_bar(3) as bar:
    letter_score_global_ref = create_letter_score_ref_allwords(words5L)
    bar()
    word_score_global_ref = create_wordfreq_score_ref(words5L)
    bar()
    word_score_nonplural_ref = identify_nonplural_words(words5L)
    bar()
print('loading complete')
print()
print()

# %%
while True:
    word_reference = words5L.copy()
    guess_list = []

    while True:
        word = input("Enter guess: ").lower()
        # 'exit' and 'restart' shortcuts within GUESS input
        if word == 'exit':
            print()
            exit()
        elif word == 'restart':
            print('wordlist refreshed!')
            print()
            break
        # print the colored squared using 'summary'
        elif word == 'summary':
            if len(guess_list) == 0:
                print('Error - no guesses made!')
                print()
                pass
            else: 
                print()
                print(str(len(guess_list))+'/6')
                for guess in guess_list:    
                    print(guess)
                print()
        
        elif word not in words5L:
            print('Error - not a valid word!')
            print()
            pass

        # valid guess word
        elif len(word)==5:
            result = input("Enter results: ").lower()
            # 'exit' and 'restart' shortcuts within RESULTS input
            if result == 'exit':
                print()
                exit()
            elif result == 'restart':
                print('wordlist refreshed')
                print()
                break
            # print the colored squared using 'summary'
            elif word == 'summary':
                print()
                for guess in guess_list:    
                    print(guess)
                print()
            
            # main working segment
            elif len(word)==len(result) and all(letter in 'gyb' for letter in result): 
                
                # print colored boxes
                guess_colors, word_print = print_box_colors(word, result)
                guess_list.append(guess_colors)
                print(guess_colors)
                print(word_print)
                print()
                
                # run main shortlisting function
                word_reference = create_shortlist(word, result, word_reference)
                print(len(word_reference), 'possible guesses:')
                print(word_reference[0:8])
                print()

            # errors on results input
            else: 
                print('Error - invalid RESULTS input!')
                print()
                pass
        
        # errors on word length
        elif len(word)>5:
            print('Error - too many letters!')
            pass
        elif len(word)<5:
            print('Error - too few letters!')
            pass

        else: pass