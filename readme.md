### Natural language Preprocessing

In Artificial intelligence handling numerical data is easy. But most of the data today is available in text. Like tweets, messages, comments, emails and documents.
NLP is purely related how the text data can be handled. The machine can only understand numbers,hence it is important to treat the text correctly and convert it to numbers so that the machine is able to process the data. 

The entire process of handling text data such that its essence, context and useful information is preserved, is called NLP

The most population library used for NLP is NLTK (natural language tool kit)


```python
!pip install nltk
```

    Requirement already satisfied: nltk in /Users/amritapanjwani/opt/anaconda3/lib/python3.9/site-packages (3.7)
    Requirement already satisfied: joblib in /Users/amritapanjwani/opt/anaconda3/lib/python3.9/site-packages (from nltk) (1.1.0)
    Requirement already satisfied: tqdm in /Users/amritapanjwani/opt/anaconda3/lib/python3.9/site-packages (from nltk) (4.64.0)
    Requirement already satisfied: click in /Users/amritapanjwani/opt/anaconda3/lib/python3.9/site-packages (from nltk) (8.0.4)
    Requirement already satisfied: regex>=2021.8.3 in /Users/amritapanjwani/opt/anaconda3/lib/python3.9/site-packages (from nltk) (2022.3.15)



```python
#Creating simple text
paragraph = """ Narendra Damodardas Modi (Gujarati: [ˈnəɾendɾə dɑmodəɾˈdɑs ˈmodiː] (listen); born 17 September 1950)[b] is an Indian politician serving as the 14th and current prime minister of India since 2014. Modi was the chief minister of Gujarat from 2001 to 2014 and is the Member of Parliament from Varanasi. He is a member of the Bharatiya Janata Party (BJP) and of the Rashtriya Swayamsevak Sangh (RSS), a right-wing Hindu nationalist paramilitary volunteer organisation. He is the longest serving prime minister from outside the Indian National Congress.
"""
```


```python
paragraph
```




    ' Narendra Damodardas Modi (Gujarati: [ˈnəɾendɾə dɑmodəɾˈdɑs ˈmodiː] (listen); born 17 September 1950)[b] is an Indian politician serving as the 14th and current prime minister of India since 2014. Modi was the chief minister of Gujarat from 2001 to 2014 and is the Member of Parliament from Varanasi. He is a member of the Bharatiya Janata Party (BJP) and of the Rashtriya Swayamsevak Sangh (RSS), a right-wing Hindu nationalist paramilitary volunteer organisation. He is the longest serving prime minister from outside the Indian National Congress.\n'



### Steps in Text Processing
- Convert a document/ paragraph into sentences
- Convert sentences to words
- Tokenize the words
- Create vectors for each word

### Libraries used:
- nltk.corpus to get stopwords
- nltk.sent_tokenize to create sentences out of a paragraph
- nltk.word_tokenize to create words out of each sentence
- nltk.stem to use stemmers and lemmatizers for tokennizing the words

### Definitions:
- stopwords: the words in any languages that do not add any context or meaning but are present as part of grammar are called stopwords. Example of stopwords: 'of'  , 'was' , 'the'


```python
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /Users/amritapanjwani/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!





    True




```python
len(stopwords.words('english'))
```




    179




```python
#Create sentences
sentences = nltk.sent_tokenize(paragraph)
```


```python
print(sentences)
```

    [' Narendra Damodardas Modi (Gujarati: [ˈnəɾendɾə dɑmodəɾˈdɑs ˈmodiː] (listen); born 17 September 1950)[b] is an Indian politician serving as the 14th and current prime minister of India since 2014.', 'Modi was the chief minister of Gujarat from 2001 to 2014 and is the Member of Parliament from Varanasi.', 'He is a member of the Bharatiya Janata Party (BJP) and of the Rashtriya Swayamsevak Sangh (RSS), a right-wing Hindu nationalist paramilitary volunteer organisation.', 'He is the longest serving prime minister from outside the Indian National Congress.']


### Cleaning the text : Step I
- Any text may consists of alphabets, numbers, special characters and symbols.
- For text analysis it is important to remove all the other characters except alphabets and numbers.
- In order to do so following loop is run to check each sentence at a time
- For each sentence regex is used to clean the text
- Next, all letters are converted to lowercase
- Finally the cleaned text is appended to the user created corpus list as shown below:


```python
#Clean text
import re
corpus = []
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z0-9]',' ', sentences[i])
    review = review.lower()
    corpus.append(review)
```


```python
corpus
```




    [' narendra damodardas modi  gujarati    n  end   d mod   d s  modi    listen   born 17 september 1950  b  is an indian politician serving as the 14th and current prime minister of india since 2014 ',
     'modi was the chief minister of gujarat from 2001 to 2014 and is the member of parliament from varanasi ',
     'he is a member of the bharatiya janata party  bjp  and of the rashtriya swayamsevak sangh  rss   a right wing hindu nationalist paramilitary volunteer organisation ',
     'he is the longest serving prime minister from outside the indian national congress ']



### Stemming and Lemmatization
- different words are written different. For example: walk is one verb, but in a text it can be present as: walking, walk , walked. Hence it is important to present such words in their root form. To do so two techniques can be used:
- Stemmpling: In this technique each word is converted to a stem form by removing few characters towards the end of each word. These stems may or may not have a meaning, but it is a faster process.
- Lemmatization: In this technique each word is converted to its root word, which is like the original simple word and has some genuine meaning. This process of reaching upto the roots requires more efforts and hence it is a time taking process.

- Both the methods have their own pros and cons. Depending on different usecases different methods are used.


### Cleaning the text : Step II (with stemming)
- In the below process for each sentence words are created using: nltk.word_tokenize
- For each word if it is not in stopwords it is processed further
- Stemmer is used to create stem of each word
- All stems are finally appended to the cleaned_text list


```python
#Apply Stemming
from nltk.stem import PorterStemmer
```


```python
#Instantiating the stemmer object
stemmer = PorterStemmer()
```


```python
cleaned_text = []
for i in corpus:
    words = nltk.word_tokenize(i)
    for word in words:
        if word not in set(stopwords.words('english')):
            print(stemmer.stem(word))
            cleaned_text.append(word)

len(cleaned_text)
```

    narendra
    damodarda
    modi
    gujarati
    n
    end
    mod
    modi
    listen
    born
    17
    septemb
    1950
    b
    indian
    politician
    serv
    14th
    current
    prime
    minist
    india
    sinc
    2014
    modi
    chief
    minist
    gujarat
    2001
    2014
    member
    parliament
    varanasi
    member
    bharatiya
    janata
    parti
    bjp
    rashtriya
    swayamsevak
    sangh
    rss
    right
    wing
    hindu
    nationalist
    paramilitari
    volunt
    organis
    longest
    serv
    prime
    minist
    outsid
    indian
    nation
    congress





    57



### Cleaning the text : Step II (with lemmatization)
- WordNetLemmatizer is used for lemmatization
- First words are created using word.tokenize
- Then each word is checked for if it is a stopword, if yes it is removed, else the further process continues
- Further lemmatizer is applied to create root of each word
- Finally all roots are appended to the list of clean_text


```python
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
```

    [nltk_data] Downloading package wordnet to
    [nltk_data]     /Users/amritapanjwani/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    [nltk_data] Downloading package omw-1.4 to
    [nltk_data]     /Users/amritapanjwani/nltk_data...
    [nltk_data]   Package omw-1.4 is already up-to-date!





    True




```python
lemmatizer = WordNetLemmatizer()
cleaned_text = []
for i in corpus:
    words = nltk.word_tokenize(i)
    for word in words:
        if word not in set(stopwords.words('english')):
            print(lemmatizer.lemmatize(word))
            cleaned_text.append(word)

len(cleaned_text)
```

    narendra
    damodardas
    modi
    gujarati
    n
    end
    mod
    modi
    listen
    born
    17
    september
    1950
    b
    indian
    politician
    serving
    14th
    current
    prime
    minister
    india
    since
    2014
    modi
    chief
    minister
    gujarat
    2001
    2014
    member
    parliament
    varanasi
    member
    bharatiya
    janata
    party
    bjp
    rashtriya
    swayamsevak
    sangh
    r
    right
    wing
    hindu
    nationalist
    paramilitary
    volunteer
    organisation
    longest
    serving
    prime
    minister
    outside
    indian
    national
    congress





    57



#### (Both methods stemming and lemmatization are shown above, any one of them can be used)
#### Compare the results of stemming and lemmatization in the above two steps

### Last step of Text Processing: Vectorization
- Converting the words into a list of 1-d arrays or vectors is called vectorization. This step is important as it transforms the words into numbers which are usable by the machine learning or deep learning algorithms. The vectors so generated are also called as <b>word-embeddings</b>

#### There are different vectorization techniques:
- Bag of words (CountVectorizer)
- tfidf (TfidfVectorizer)
- Word2Vec


#### BOW summary: 
- This technique provides all the unique words present in the training dataset and hence it is called bag of words. The list of words is ordered as per the frequency or importance of the word in the training dataset. These words are considered as features. (Specific terminology used in Data Science). 

- The words are always listed in order of their importance in the corpus.For each sentence or document we get a list of zeros and frequencies based on whether a word is present in that sentence or not. 

- Hence if there are total n words in the training dataset and m sentences , we get a m x n matrix of zeros and frequencies. This matrix is called sparse matrix. Like this the entire text is converted to numbers and vectorization happens.

#### Parameters of CountVectorizer:
- analyzer: {word, character}This parameter lets the countvectorizer know if to consider a word as a feature or a character. By default it considers a 'word' as a feature
- ngram_range: (min, max) , this tells if we need to take only one word as a feature or group of 2 or three words together as one feature. For exampl: "medical document" is a phrase with two words, it can be considered as 1 single feature by countvectorizer if ngram_range is mentioned as(1,2). This allows the countvectorizer to take all unigrams and bigrams as features.
- token_patthern: it takes a string value, it tells what kind of tokens can be considered. Default token pattern to denote a word is: r”(?u)\b\w\w+\b”
- stop_words: it takes a string which denotes language like 'english'. This tells countvectorizer to remove all stop_words
- lowercase: {true/ false} this tells the countvectorizer to convert all text into lower case before tokenzing. Default value is 'true'.
- max_feautres: if there is large text we can let the countvectorizer select only top certain number of features based on their importance or frequency of occurrence in the training dataset. It accepts integer values.
- vocabulary: this results in a mapping of each word(feature) with its index. The list of vocabulary displays such that most important or most occurring features shows up on top.
- binary: (true/ false): this tells the countvectorizer if it has to create a matrix with frequency of each feature or not. If frequency is required binary: false. This is the default value. If binary: true, this will count only presence or absence of a word in a sentence. Presence is captured as 1 and absence is captured as 0. Even if a word occurs several number of times it a sentence it will be reflected as 1 only.





```python
#Working on bag of words:
# We use the library countvectorizer to work on bag of words(BOW)

from sklearn.feature_extraction.text import CountVectorizer

```

#### Steps involved:
- First clean the text with all other irrelevant characters
- Convert all characters to lower case
- split the sentences to words
- convert the words to their root form using lemmatizer while removing the stopwords
- Finally join all the lemmatized roots and append to corpus_new list
- After all the above "cleaning steps" its time to convert these words to vectors
- cv is the CountVectorizer object, fit_transform it to the corpus_new to generate vectors for all the words


```python
corpus_new = []
cv = CountVectorizer()
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]',' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus_new.append(review)
```


```python
#Creating the vectors for all words in all sentences
x = cv.fit_transform(corpus_new)
```


```python
# Check all words using vocabulary_
cv.vocabulary_
```




    {'narendra': 20,
     'damodardas': 6,
     'modi': 19,
     'gujarati': 9,
     'end': 7,
     'mod': 18,
     'listen': 14,
     'born': 2,
     'september': 33,
     'indian': 12,
     'politician': 28,
     'serving': 34,
     'th': 37,
     'current': 5,
     'prime': 29,
     'minister': 17,
     'india': 11,
     'since': 35,
     'chief': 3,
     'gujarat': 8,
     'member': 16,
     'parliament': 26,
     'varanasi': 38,
     'bharatiya': 0,
     'janata': 13,
     'party': 27,
     'bjp': 1,
     'rashtriya': 30,
     'swayamsevak': 36,
     'sangh': 32,
     'right': 31,
     'wing': 40,
     'hindu': 10,
     'nationalist': 22,
     'paramilitary': 25,
     'volunteer': 39,
     'organisation': 23,
     'longest': 15,
     'outside': 24,
     'national': 21,
     'congress': 4}



#### In the above scenario all these numbers are the index, not the frequency


```python
#There are total 4 sentences in corpus_new
len(corpus_new)
```




    4




```python
# x is the sparse matrix of 4 sentences and 41 unique words in the dataset
x.shape
```




    (4, 41)




```python
#Checking the first row of the matrix for sentence one
x[0].toarray()
```




    array([[0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 2, 1, 0,
            0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0]])



#### Disadvantages of Bag Of Words technique of vectorization:
- It works great for smaller datsets with smaller bag of words. But as the dataset increases, the computation cost of it increases.

- Out of vocabulary: if a word is not there while creating BOWs, then if the word appears later during the testing stage, it cannot be identified or treated by the model.

- Lack of semantic meaning: The presence of word "not": the entire sentence is same but "not" is the only word that is new. In this case, the assignment of 0 and 1 would be exactly same for both sentences, which makes it look that two sentences are similar. These vectors are very close to each other. But infact by just one word "not" the two sentences are totally opposite to each other.This is called lack of semantic meaning. It does not tell which word to focus more on.

### TFIDF Vectorizer:
##### To give importance to some words in a sentence to get their semantic meaning we use TFIDF. Aim: some of the words that are rare in that sentence must be given more weightage.

#### Step for TFIDF calculation:
- TF = number of repetitions of word in sentence/ total number of words in sentence

- IDF = LOGe(total number of sentences/ number of sentences containing the word)

- TFIDF(word1 in sentence1) = TF(word1 in sentence1) * IDF(word1 in sentence1)

Like this TFIDF of all words in all sentences is assigned a score. This helps capture semantic information from the sentences. Frequent words have smaller tfidf score and rare words have higher tfidf score
##### Advantages: intuitive, semantic, word importance is getting captured
##### Disadvantage: with large data sparisty would be there , oov(out of vocabulary)

#### Note on ngrams_range attribute in countvectorizer:
- ngram_range = (3,3) it will select triagrams.
- ngram_range = (2,3) it will capture all bigrams and triagrams.
- ngram_range = (1,3) it will select unigram , biagrams and triagrams.

#### Noted: Tfidfvectorizer has exact same parameters as CountVectorizer


```python
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
```


```python
x=tf.fit_transform(corpus_new)
```


```python
corpus_new[0]
```




    'narendra damodardas modi gujarati n end mod modi listen born september b indian politician serving th current prime minister india since'




```python
x[0].toarray()
```




    array([[0.        , 0.        , 0.23729913, 0.        , 0.        ,
            0.23729913, 0.23729913, 0.23729913, 0.        , 0.23729913,
            0.        , 0.23729913, 0.18708936, 0.        , 0.23729913,
            0.        , 0.        , 0.15146496, 0.23729913, 0.37417871,
            0.23729913, 0.        , 0.        , 0.        , 0.        ,
            0.        , 0.        , 0.        , 0.23729913, 0.18708936,
            0.        , 0.        , 0.        , 0.23729913, 0.18708936,
            0.23729913, 0.        , 0.23729913, 0.        , 0.        ,
            0.        ]])




```python
x.shape
```




    (4, 41)



word embeddings is a technique that converts words into vectors
Two methods of word emebeddings:
- 1.Count/frequency of words (BOW, TFIDF, OHE)
- 2.Deep Learning trained model(word2vec: cbow, skipgram)

we have already covered the first approach now switching to word2vec
This requires understanding of ANN, loss functions and all

#### WORD2VEC:
It is a word_embedding or vectorization technique that is based on neural networks (deep learning). In word2vec each word is converted to a vector of <b>"LIMITED DIMENSIONS"</b>. It will try to reduce sparsity , this means you would not find any 0's or 1's. It ensures that the semantic meaning is maintained. It would assign vectors such that if these vectors are plotted in multidimensional space words with similar meaning will be plotted close to each other.

### There are two word2vec architecures: CBOW and skipgram

#### Some basic terminologies:
- CBOW ( continuous bag of words)
- corpus: means group of sentences
- window size: number of continuous words you take
- window size is a hyperparameter, the bigger the better the model
- <b>Example:</b> if suppose window size is 5 and you consider the third word as the output word then the remaining 4 words on either sides are considered as the input or context words.

- <b>Practical example:</b> John is working as consultant.
- output word: "working"  
- input words: "john is as consultant"

- Note: target word is always taken in middle to get input context on both sides of the target

- <b>Aim of word2vec:</b> take the text and generate words, such that semantic meaning is captured

- These independent features and the target features and their corresponding BOW representation is given to the fully connected layer (architecture of neural networks is used)

#### Brief description
Different weights are initialized, then it passes through softmax layer and loss is calculated. 
So as the neurons move from input through hidden layer and output this is called forward propagation.
Once it passes through softmax layer, loss is calculated and weight are updated this is called backward propagation.
This continues till the time minimum loss is obtained the weights assigned at each transformation combined together results in embeddings
This is the process of Continuous Bag of Words method (CBOW)

<b>Note:</b> Basics of ANN are important to understand the indepth architecture of word2vec.

<b>Skip gram:</b> is same as CBOW, just input and output are reversed. Both CBOW and skipgram are good, if larger data is there use SKIPGRAM.

<b>Python library used: gensim</b>


```python
!pip install gensim
```

    Requirement already satisfied: gensim in /Users/amritapanjwani/opt/anaconda3/lib/python3.9/site-packages (4.1.2)
    Requirement already satisfied: smart-open>=1.8.1 in /Users/amritapanjwani/opt/anaconda3/lib/python3.9/site-packages (from gensim) (5.1.0)
    Requirement already satisfied: scipy>=0.18.1 in /Users/amritapanjwani/opt/anaconda3/lib/python3.9/site-packages (from gensim) (1.7.3)
    Requirement already satisfied: numpy>=1.17.0 in /Users/amritapanjwani/opt/anaconda3/lib/python3.9/site-packages (from gensim) (1.21.5)



```python
import gensim
```


```python
from gensim.models import word2vec
```

<b>Generally pretrained embeddings are used for word2vec. You can also train your own data and create word embeddings. Here pretrained embeddings are used "word2vec-google-news-300" This means word2vec embeddings are created by getting trained on google news data and these embeddings are of the dimension 300.</b>


```python
import gensim.downloader as api
wv = api.load('word2vec-google-news-300')
vec_sun = wv['sun']
```


```python
#below is the embedding for the word "king" There are 300 elements in the below array
vec_sun
```




    array([-0.02600098,  0.16894531,  0.03955078, -0.03149414,  0.07470703,
           -0.14160156,  0.21191406, -0.30078125, -0.06347656,  0.10302734,
            0.22265625,  0.12304688,  0.02416992,  0.01831055, -0.27929688,
            0.26757812, -0.06640625, -0.02685547,  0.30664062, -0.15332031,
           -0.23828125,  0.33984375, -0.02709961, -0.1484375 , -0.10839844,
           -0.08496094,  0.20117188,  0.1796875 , -0.03039551, -0.09375   ,
           -0.36523438, -0.04174805, -0.25390625, -0.16699219, -0.27539062,
           -0.12890625, -0.15820312, -0.36328125,  0.00177765,  0.06689453,
            0.33789062, -0.34570312,  0.12060547, -0.00346375,  0.08300781,
           -0.22558594, -0.20507812, -0.0071106 , -0.11083984, -0.12402344,
            0.34960938,  0.42578125,  0.32421875,  0.20117188,  0.04467773,
           -0.06591797,  0.06396484,  0.13183594, -0.07861328,  0.2421875 ,
           -0.40625   , -0.17480469,  0.02270508, -0.08789062,  0.27148438,
           -0.06787109, -0.32226562, -0.12255859, -0.21289062,  0.11816406,
            0.25585938,  0.05493164,  0.11376953, -0.06591797, -0.48046875,
            0.23632812,  0.09619141, -0.09277344, -0.12597656, -0.07666016,
            0.30078125, -0.0402832 ,  0.08789062, -0.18359375, -0.32617188,
            0.23925781, -0.01251221,  0.02685547,  0.14550781,  0.16113281,
           -0.203125  , -0.12207031, -0.24511719, -0.01574707, -0.29296875,
            0.25      ,  0.29296875, -0.34179688,  0.20703125, -0.24414062,
           -0.10253906, -0.140625  ,  0.15039062, -0.04663086, -0.09326172,
            0.01464844,  0.02648926,  0.06835938, -0.16210938, -0.14355469,
            0.02539062,  0.15039062,  0.1484375 ,  0.14355469,  0.4765625 ,
            0.14648438,  0.23632812, -0.14746094, -0.3984375 ,  0.13085938,
           -0.18359375,  0.08935547, -0.03588867, -0.07470703,  0.05834961,
           -0.04492188,  0.00668335,  0.13867188,  0.00366211,  0.14355469,
           -0.09960938, -0.15136719, -0.15722656, -0.20703125, -0.08789062,
            0.1875    ,  0.07519531,  0.02526855, -0.00970459,  0.08251953,
            0.19433594,  0.09667969, -0.03613281, -0.140625  ,  0.27539062,
            0.16796875,  0.33007812, -0.03613281,  0.01367188, -0.18457031,
            0.03881836,  0.02624512, -0.08154297, -0.00212097, -0.02636719,
           -0.06005859,  0.18554688, -0.53515625,  0.21875   , -0.04541016,
           -0.20800781,  0.02124023, -0.07568359,  0.10009766, -0.05786133,
           -0.06884766, -0.21386719,  0.14160156,  0.12792969,  0.09423828,
           -0.17480469,  0.42773438, -0.00430298,  0.29101562, -0.20703125,
           -0.08447266, -0.01599121, -0.02392578, -0.20019531, -0.47265625,
           -0.29882812, -0.28125   , -0.10253906, -0.13671875, -0.36523438,
           -0.13085938, -0.1796875 , -0.19140625,  0.22460938, -0.01287842,
            0.08691406,  0.08349609,  0.16503906,  0.10498047, -0.05981445,
            0.51171875, -0.13867188, -0.05517578,  0.18164062, -0.07128906,
            0.40820312, -0.17773438, -0.12158203, -0.2890625 ,  0.16894531,
           -0.02722168,  0.10205078,  0.06835938, -0.20507812, -0.08935547,
           -0.25      ,  0.36914062, -0.05737305, -0.05639648,  0.16503906,
           -0.1640625 , -0.16503906,  0.19433594, -0.17578125, -0.04736328,
            0.23925781, -0.06152344,  0.3671875 ,  0.09277344, -0.30273438,
            0.04492188,  0.15136719,  0.08886719,  0.1640625 ,  0.02294922,
           -0.15917969, -0.10888672, -0.06884766, -0.16503906,  0.12695312,
           -0.0189209 , -0.02001953,  0.00759888,  0.12402344, -0.02392578,
           -0.12011719, -0.07421875,  0.10107422, -0.3515625 ,  0.09228516,
           -0.15527344, -0.01025391,  0.08984375,  0.078125  ,  0.08007812,
            0.15820312, -0.16503906, -0.08056641,  0.32226562, -0.13671875,
           -0.20117188,  0.07958984,  0.03857422, -0.28320312, -0.04589844,
           -0.01867676, -0.45703125, -0.234375  , -0.11621094,  0.2109375 ,
            0.09814453, -0.03686523, -0.10205078,  0.02709961, -0.17089844,
           -0.07128906,  0.07275391,  0.01586914,  0.12695312, -0.05908203,
            0.0088501 , -0.26367188, -0.31445312, -0.12158203,  0.12207031,
            0.14941406,  0.203125  ,  0.18066406,  0.09619141, -0.06298828,
           -0.26757812,  0.03735352,  0.38671875, -0.21582031,  0.2109375 ,
            0.21191406,  0.34765625, -0.0559082 ,  0.14453125,  0.01239014,
            0.32421875, -0.00254822, -0.00653076,  0.22070312,  0.13867188],
          dtype=float32)




```python
vec_sun.shape
```




    (300,)




```python
wv['moon']
```




    array([-0.03857422,  0.18945312,  0.20605469,  0.171875  ,  0.05419922,
           -0.22460938,  0.4140625 , -0.35351562,  0.21484375,  0.05688477,
            0.18847656,  0.15917969, -0.06225586, -0.05029297, -0.00622559,
            0.25585938, -0.09423828,  0.00491333,  0.29101562, -0.06030273,
           -0.39453125, -0.04077148,  0.16308594, -0.05444336,  0.02514648,
            0.07861328, -0.01361084,  0.02746582,  0.43164062,  0.13769531,
           -0.41992188, -0.17578125, -0.1796875 ,  0.04174805, -0.08691406,
           -0.11279297,  0.01013184, -0.01397705,  0.17773438,  0.01275635,
            0.10351562,  0.04711914,  0.41015625,  0.28515625,  0.10791016,
           -0.17089844, -0.0559082 , -0.30859375,  0.04321289, -0.03076172,
           -0.01202393,  0.24316406, -0.10253906,  0.15039062, -0.12451172,
           -0.34765625,  0.43554688, -0.0534668 , -0.04589844, -0.04223633,
           -0.0324707 , -0.23144531,  0.11083984,  0.003479  , -0.21777344,
            0.07226562, -0.11669922,  0.04516602,  0.13183594,  0.09130859,
            0.04052734, -0.296875  ,  0.04663086,  0.14160156, -0.38476562,
            0.05322266,  0.11035156,  0.16601562, -0.13867188,  0.234375  ,
            0.13964844, -0.11181641,  0.2734375 , -0.43945312, -0.03369141,
           -0.15722656,  0.15527344, -0.39257812,  0.35546875,  0.05053711,
            0.05859375,  0.00424194, -0.22070312, -0.29492188,  0.04418945,
           -0.12158203,  0.37304688, -0.03930664,  0.37890625, -0.03198242,
           -0.10351562, -0.06787109,  0.19628906, -0.00276184,  0.05664062,
            0.23925781, -0.05249023, -0.00132751, -0.05322266, -0.10253906,
           -0.17871094,  0.10400391,  0.12988281,  0.15625   ,  0.04467773,
            0.05688477,  0.01037598, -0.02514648, -0.22070312,  0.02722168,
           -0.13769531, -0.23730469, -0.01361084, -0.01165771, -0.14257812,
           -0.2890625 ,  0.00946045,  0.14355469,  0.30078125,  0.05126953,
           -0.06347656, -0.01177979, -0.20996094,  0.08984375, -0.01501465,
            0.25195312, -0.04711914, -0.04663086,  0.25      ,  0.421875  ,
           -0.078125  , -0.15136719, -0.2421875 ,  0.30078125,  0.15820312,
            0.08398438, -0.05371094, -0.12597656,  0.11376953, -0.07666016,
            0.24511719, -0.05102539, -0.12402344,  0.01312256,  0.00692749,
           -0.41601562,  0.16503906,  0.0546875 ,  0.10058594, -0.0291748 ,
           -0.3671875 ,  0.07226562,  0.25585938, -0.14550781, -0.00054169,
           -0.28515625,  0.25195312,  0.13671875, -0.19726562,  0.10400391,
           -0.04467773,  0.33007812,  0.09619141,  0.34960938, -0.00610352,
           -0.22265625, -0.13085938, -0.13378906, -0.12890625, -0.2421875 ,
           -0.35546875,  0.01708984, -0.19433594, -0.20019531, -0.06835938,
           -0.171875  , -0.15527344,  0.10009766,  0.06445312, -0.10888672,
            0.07958984,  0.04443359, -0.05566406, -0.10693359, -0.36328125,
            0.296875  ,  0.40625   , -0.0222168 ,  0.07910156, -0.00878906,
            0.0324707 , -0.13378906, -0.25390625, -0.26953125,  0.19433594,
           -0.06982422,  0.01525879, -0.13085938, -0.31445312,  0.16210938,
           -0.24804688,  0.1875    ,  0.09082031, -0.14941406,  0.18261719,
            0.04541016,  0.0859375 ,  0.11816406,  0.01806641, -0.2109375 ,
            0.34765625,  0.07666016,  0.44921875, -0.05004883,  0.06933594,
            0.12792969,  0.0123291 ,  0.08642578,  0.33984375,  0.08203125,
            0.37890625, -0.19238281,  0.12792969,  0.0703125 ,  0.20800781,
           -0.16015625, -0.27929688, -0.26171875, -0.05761719, -0.27929688,
            0.14453125, -0.17382812,  0.04370117, -0.15039062, -0.00323486,
           -0.28710938,  0.07910156, -0.03710938,  0.03222656,  0.01470947,
           -0.18945312, -0.27148438,  0.25976562,  0.28710938, -0.12597656,
           -0.30859375,  0.15039062, -0.265625  , -0.05029297, -0.23632812,
           -0.06542969, -0.42578125, -0.19042969, -0.15234375,  0.23828125,
            0.12255859, -0.2578125 , -0.23828125,  0.17285156,  0.06494141,
           -0.08447266, -0.06103516,  0.34765625,  0.09619141,  0.07519531,
            0.20507812, -0.23632812, -0.08056641, -0.14550781, -0.11523438,
            0.00805664,  0.30859375,  0.05371094,  0.17675781,  0.33203125,
           -0.13867188,  0.09960938, -0.00823975, -0.00436401,  0.06005859,
            0.05664062,  0.17089844, -0.2578125 , -0.03979492, -0.10107422,
            0.00598145, -0.02539062, -0.3359375 ,  0.1640625 , -0.25390625],
          dtype=float32)



#### Cosine similarity: 
- It is defined as angle between two vectors in n-dimensional space. 
- Cosine distance = 1 - cosine similarity
- So vectors created by word2vec are evaluated using cosine similarity.
- The score varies between 0 and 1. 0 indicates dissimilarity and 1 indicates similarity.


```python
#This returns list of tuples with words similar to SUN
wv.most_similar('sun')
```




    [('sunlight', 0.7269680500030518),
     ('sun_rays', 0.6871297955513),
     ('sunshine', 0.6767958402633667),
     ('sunrays', 0.6644463539123535),
     ('noonday_sun', 0.6227595806121826),
     ('rays', 0.601547360420227),
     ('suns_rays', 0.5943775773048401),
     ('dried_tomato_basil', 0.582387387752533),
     ('sun_shining', 0.5802728533744812),
     ('UV_rays', 0.574922502040863)]




```python
# This method takes two words and returns the cosine similarity between those two words
wv.similarity('python','anaconda')
```




    0.52073956



#### Observation:
The below operation basically shows how the context of words is retained by word2vec. Do a simple mathematical operation between the vectors of boy, lad and girl and the resultant vector is closest to the word girl. 
This shows that word2vec vectorization process retains the semantic meaning of the words.


```python
vec = wv['boy']-wv['lad']+wv['girl']
wv.most_similar([vec])[0]
```




    ('girl', 0.7991686463356018)


