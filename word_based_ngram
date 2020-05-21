class NGram(object):

    def __init__(self):
        self.n = None
        self.ngramprobability = None

    
    def train(self,m,text):
        
        normalizedtext = text.lower()
        #removing unnecessary punctuations
        b = re.sub(r"[^a-z']+", ' ', normalizedtext)
        #self.wordlist = list(b.split(" "))[:-1]
        words = nltk.wordpunct_tokenize(b)
        paddedwords = list(pad_sequence(words,pad_left=True,left_pad_symbol = "<s>",pad_right= True,right_pad_symbol="</s>",n = m))
        for elements in paddedwords:
            self.wordlist = ['<UNK>' if paddedwords.count(elements)==1 else elements for elements in paddedwords]
        #print(self.wordlist)
        self.allgrams = []
        for i in range(len(self.wordlist)-(m-1)):
            self.allgrams.append(self.wordlist[i:i+m])
        while m > 1:
            m = m-1
            for i in range(len(self.wordlist)-(m-1)):
                self.allgrams.append(self.wordlist[i:i+m])
        #self.allgrams = allgrams
        
        self.uniquegrams = [list(x) for x in set(tuple(x) for x in self.allgrams)]
        #print(self.uniquegrams)
        
        df_unigrams = []
        df_allgrams = []
        for grams in self.uniquegrams:
            if len(grams) == 1:
                unigram = []
                unigram_prob = self.wordlist.count(grams[0])/len(self.wordlist)
                unigram.append(grams[0])
                unigram.append(unigram_prob)
                df_unigrams.append(unigram)
            else:
                ngram_probability = self.allgrams.count(grams) / self.allgrams.count(grams[:-1])
                entry = []
                for word in grams:
                    entry.append(word)
                entry.append(ngram_probability)
                df_allgrams.append(entry)
            finallist = df_unigrams + df_allgrams
            d = defaultdict(list)
            for i in finallist:
                d[len(i)].append(i)
            self.out = [d[i] for i in sorted(d)]
        #print(self.out)
        
        #ngralists = pd.DataFrame(lists) for lists in out
        df_lists = [pd.DataFrame(lists) for lists in self.out]
        self.ngramprobability = df_lists
        
        return df_lists

    
    def save(self,path):
        pickle_out = open(path,"wb")
        pickle.dump(self.ngramprobability,pickle_out)
        return pickle_out.close() 
    
    def load(self,path):
        with open(path,'rb') as read_file:
            df_lists = pickle.load(read_file)
        return df_lists
    
    def word_predict(self,n,inputgram):
        tuples = list({tuple(a) for b in self.out for a in b})
        return [element[-2:] for element in tuples if len(element) == n+1 if all(a == b for a, b in zip(element, inputgram))]
        
#     def word_predict(self,w1):
#         cond = self.ngramprobability.iloc[:,:-2].isin(w1).all(axis=1).to_numpy()
#         word = self.ngramprobability.iloc[cond]
#         tuples = [tuple(x[-2:]) for x in word.to_numpy()]
#         return tuples  
    
#     def get_probability(self,ngram):
#         cond = self.ngramprobability.iloc[:,:-1].isin(ngram).all(axis=1).to_numpy()
#         prob = self.ngramprobability.iloc[cond]
#         probability = [tuple(x) for x in prob.to_numpy()]
#         return probability
    
    def stupid_backoff(self,words):
        
        for i in words:
            words = ['<UNK>' if i not in self.wordlist else i for i  in words]
        #print(words)
        
        if len(words) == 1:
            probability =  [j[-1] for i in self.out for j in i if j[:-1] == words][0]
            return probability
    
        elif words in self.uniquegrams:
            probability =  [j[-1] for i in self.out for j in i if j[:-1] == words][0]
            return probability
        else: 
            return self.stupid_backoff(words[1:])*0.4
        
            

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import nltk,re,pprint,collections,json,csv
    from collections import defaultdict
    from nltk.util import pad_sequence
    
    # train bi_gram
    bi_gram = NGram()
    text = "This is a sample this is interesting and I like it This is a sample this is interesting and I like it and not bad at all but I don't like it I dont know why?Do you know why ?."
    #df = pd.DataFrame({"samples":["this is a test", "this is interesting"]})
    bi_gram.train(3,text)
    #print(b)
    
    c = bi_gram.stupid_backoff(["<s>","this",'sample','problem','is','interesting','</s>'])
    print("probability of given sequence is:",c)
    
    # generate some random sentences
    num_words = 5
    n_gram = 2 #put n_gram as bigram = 2,trigram = 3...upto ngram
    text = "this"
    prefix = (text,)

    for i in range(num_words):
        next_words = bi_gram.word_predict(n_gram,prefix)
        #print(next_words)
        ngramprobability = [entry[-1] for entry in next_words]
        rand_index = np.random.choice(len(ngramprobability), 1, p=ngramprobability)[0]
        next_word = next_words[rand_index][0] 
        #print(next_word)

        text += " " + next_word
        #print(text)
        prefix = (next_word,)

    # print generated text
    print(text)
