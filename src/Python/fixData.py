import json
import os
from transformers import BertTokenizer
import fasttext
import re
import numpy as np
import pickle
#

####

step1 = False
step2 = False
step3 = True

#####


path = "\\".join(os.getcwd().split("\\")[:-3]) + "\\data\\data" #Just make the path to the data folder :)

filesExist = True
try:
    files = os.listdir(path)
except:
    filesExist = False

print("There are:", len(files), "files")


ft_model = fasttext.load_model("\\".join(os.getcwd().split("\\")[:-3]) + "\\lid.176.bin") #Import of the FastText model to indentify the language


#region ORGINIZING THE DATA 

def correctText(text: str, remove = True):
    corr = text

    #corr = corr.replace("t.ex.", "till exempel")
    #corr = corr.replace("t.ex", "till exempel")

    if remove:
        for sep in ["-", "!", "&", "%", "(", ")", "]", "[", "{", "}", "$", ":",";", "®"]: #Removal
            corr = corr.replace(sep, "")
    #else:
    #    for sep in [".",  ",", "/", "?"]: #Change
    #        corr = corr.replace(sep, " " + sep + " ")

    #corr = corr.replace(". net", ".net")
    return corr


def getText(path: str, i: int): 
    '''
    For getting the texts in one file, and clean the data.
    '''
    try:
        if (i % 1000) == 0:
            print("Text nr", i, "is read")

        with open(path, "r", encoding="utf-8") as jFile:
            data = json.load(jFile)

        annons = data["platsannons"]["annons"]
        
        text_no_link = [t for t  in annons["annonstext"].lower().strip().split() if "http" not in t] #remove all adresses
        text_no_mail = " ".join([t for t  in text_no_link if  "@" not in t])

        ## Removal of unwanted letters

        text = text_no_mail.replace("\n", " ").replace("\t", " ").replace("##", "")
        text = correctText(text, True)
        #3text = correctText(text, False)

        header = annons["annonsrubrik"]

        prediction = ft_model.predict(text) #Use model to indentify swe or eng language
        
        #Only start with handeling swedish stuff first and probability of 80% and higher
        if (len(prediction[0]) == 1) & (prediction[0][0] == '__label__sv') & (prediction[1][0] > 0.8):
            #Clean some text
            index = text.find("KONTAKTINFORMATION")

            return text[:index] #remove exess contact information and return the text

    except:
        print("error file", i, "of file", len(files))


if filesExist & step1:
    texts = [getText(path + "\\" + fi, i) for i, fi in enumerate(files[1:])]

    print("The amount of swedish texts are:", len(texts), "out of", len(files), "files")
    ##Write it all to file
    try:
        with open("texts.pkl", "wb") as txt:
            pickle.dump(texts, txt)
            #for t in texts:
            #    if t:
            #        txt.write(t)
            #        txt.write(";")
    except UnicodeEncodeError as e:
        print("error with text:\n", t)
#endregion


#region Label the text
'''
O: Not anything of importance-tag
S: A skill-tag
L: A language-tag
'''



if not step1 & step2:
    with open("texts.pkl", "rb") as txt:
        texts = pickle.load(txt)



with open("knownTagsLang.txt", "r", encoding="utf-8") as txt:
    lang = txt.read().split("\n")


with open("knownTagsSkills.txt", "r", encoding="utf-8") as txt:
    skills = txt.read().split("\n")


def getTag(token: str) -> str:
    ret = "O"
    if token in lang:
        ret = "L"
    elif token in skills:
        ret = "S"

    return ret


def tagText(text: str, i: int):
    try:
        if (i % 1000) == 0:
            print("Tagged", i, "texts")

        #tags_pre = [getTag(token) for token in text.split()]
        #text = f'[CLS] {text} [SEP]'
        tokens = tokenizer.tokenize(text) #tokenize the text
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        tags = [getTag(token) for token in tokens]
        #print(len(tags), len(tokens))
        '''
        #fix the tags so that it matches the new tokens
        token_tmp = [] #just a Temporart store for tokens
        lag = 1
        for i in range(len(tokens)):
            if "##" in tokens[i]: #If we have a ## in a word, then look at the previous word and merge them for now.
                token_tmp[i - lag] = token_tmp[i - lag] +  tokens[i]
                lag += 1
            else:
                token_tmp.append(tokens[i])

        
        tags = [] #Get more tags
        for tag, token in zip(tags_pre, token_tmp):
            nrSplits = token.count("##")
            if tag == "L":
                print(token)
            tagMult = [tag] * (nrSplits + 1) #Multiply the tags according to how many splits the word had
            tags += tagMult
        '''
        return [tags, tokens, token_ids]
    
    except:
        print("Error at file nr", i)

    








if step3 & step2:
    print("Starting to tag the text:")

    pretrained_model_name = 'af-ai-center/bert-base-swedish-uncased'
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name, do_lower_case=False)
    tokenizer.add_tokens(lang) # Add language and skills to library so that we do not split these
    tokenizer.add_tokens(skills)

    tagTexts = [tagText(text, i) for i,text in enumerate(filter(None, texts))]


    tags = {f"T{i}": " ".join(pair[0]) for i, pair in enumerate(tagTexts)}
    textsProcessed = {f"T{i}": " ".join(pair[1]) for i, pair in enumerate(tagTexts)}
    texts_ids = {f"T{i}": pair[2] for i, pair in enumerate(tagTexts)}

    with open("tagsPreProcessed.pkl", "wb") as pFile:
        pickle.dump(tags, pFile) 

    with open("textsPreProcessed.pkl", "wb") as pFile:
        pickle.dump(textsProcessed, pFile) 

    with open("texts_idsPreProcessed.pkl", "wb") as pFile:
        pickle.dump(texts_ids, pFile) 

elif step3 & (not step2):
    print("Loading the pickeled data.\n")
    
    with open("tagsPreProcessed.pkl", "rb") as pFile:
        tags = pickle.load(pFile) 
    '''
    with open("textsPreProcessed.pkl", "rb") as pFile:
        pickle.load(pFile) 

    with open("texts_idsPreProcessed.pkl", "rb") as pFile:
        pickle.load( pFile) 
    '''


#endregion


#region Final statistics


nrS = []
nrL = []
nrO = []
iter_items = list(tags.keys())
for tagLabel in iter_items:
    tagCollection = tags[tagLabel]
    nrS.append(tagCollection.count("S"))
    nrL.append(tagCollection.count("L"))
    nrO.append(tagCollection.count("O"))


print("\nThere are", len(iter_items), "texts")
print("\n{:>5} {:>13} | {:<10}".format("Tag", "Count", "Average"))
print("---------------------------------")
print("{:>5} {:>13} | {:<10}".format("L-tag", sum(nrL),  round(np.mean(nrL), 3)))
print("{:>5} {:>13} | {:<10}".format("S-tag", sum(nrS),  round(np.mean(nrS), 3)))
print("{:>5} {:>13} | {:<10}".format("O-tag", sum(nrO),  round(np.mean(nrO), 3)), "\n")







#endregion






