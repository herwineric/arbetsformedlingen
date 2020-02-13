import json
import os
from transformers import BertTokenizer
import fasttext
#

path = "\\".join(os.getcwd().split("\\")[:-3]) + "\\data\\data" #Just make the path to the data folder :)

filesExist = True
try:
    files = os.listdir(path)
except:
    filesExist = False

print("There are:", len(files), "files")


ft_model = fasttext.load_model("\\".join(os.getcwd().split("\\")[:-3]) + "\\lid.176.bin") #Import of the FastText model to indentify the language

#region ORGINIZING THE DATA 
def getText(path: str, i: int): 
    '''
    For getting the texts in one file, and a light clean of the data.
    '''
    try:
        if (i % 1000) == 0:
            print("Text nr", i, "is read")

        with open(path, "r", encoding="utf-8") as jFile:
            data = json.load(jFile)

        annons = data["platsannons"]["annons"]

        text_no_link = " ".join([txt for txt  in annons["annonstext"].split() if "http" not in txt]) #remove all adresses

        text = text_no_link.replace("\n", " ").replace("\t", " ").replace(";", "").replace(":", "").lower().strip()
        header = annons["annonsrubrik"]

        prediction = ft_model.predict(text) #Use model to indentify swe or eng language
        
        #Only start with handeling swedish stuff first and probability of 75% and higher
        if (len(prediction[0]) == 1) & (prediction[0][0] == '__label__sv') & (prediction[1][0] > 0.75):
            #Clean some text
            index = text.find("KONTAKTINFORMATION")

            return text[:index] #remove exess contact information and return the text

    except:
        print("error file", i, "of file", len(files))


if filesExist:
    texts = [getText(path + "\\" + fi, i) for i, fi in enumerate(files[1:10])]

    print("The amount of swedish texts are:", len(texts), "out of", len(files), "files")
    ##Write it all to file
    try:
        with open("texts.txt", "w", encoding="utf-8") as txt:
            for t in texts:
                if t:
                    txt.write(t)
                    txt.write(";")
    except UnicodeEncodeError as e:
        print("error with text:\n", t)
#endregion


#region Tokenizer

### Tokeinze region
pretrained_model_name = 'af-ai-center/bert-base-swedish-uncased'

tokenizer = BertTokenizer.from_pretrained(pretrained_model_name, do_lower_case=False)


tokens = tokenizer.tokenize(texts[0])




#endregion