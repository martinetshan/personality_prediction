from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse

import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import re
import random
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


app = FastAPI()

# static

# Save Introvert vs Extrovert model
file_name_i = "xgb_introv.pkl"

# Save Sensing vs Intuition model
file_name_s = "xgb_sens.pkl"

# Save Thinking vs Feeling model
file_name_t = "xgb_think.pkl"

# Save Perceiving vs Judging model
file_name_p = "xgb_perc.pkl"

# Saving the CountVectorizer object
file_name_cv = "cv.pkl"

@app.get("/accuracy/image", status_code=200)
async def get_personality_image(image_name: str):
    try:
        if os.path.exists(f'{image_name}.png'):
            return FileResponse(f"{image_name}.png", media_type="image/png", filename=f"{image_name}.png")
        results = pd.read_csv("prediction_result.csv")
        results.plot.bar(figsize=(15, 6))
        plt.xticks([0,1,2,3],["Introvert", "Sensing", "Thinking", "Perceiving"], rotation='horizontal')
        plt.xlabel('Personality Type', fontsize=14)
        plt.title("Accuracy Scores for different models")
        plt.savefig(f'{image_name}.png')
        return FileResponse(f"{image_name}.png", media_type="image/png", filename=f"{image_name}.png")
    except Exception:
        return JSONResponse(
        status_code=404,
        content={"message": "File Not Found"},
    )


async def plot_wordcloud(text, bag_name):
    nltk.download('stopwords')
    stop=set(stopwords.words('english'))

    def _preprocess_text(text):
        corpus=[]
        stem=PorterStemmer()
        lem=WordNetLemmatizer()
        for news in text:
            words=[w for w in word_tokenize(news) if (w not in stop)]

            words=[lem.lemmatize(w) for w in words if len(w)>2]

            corpus.append(words)
        return corpus
    
    corpus=_preprocess_text(text)
    
    wordcloud = WordCloud(
        background_color='white',
        stopwords=set(STOPWORDS),
        max_words=100,
        max_font_size=30, 
        scale=3,
        random_state=1)
    
    wordcloud=wordcloud.generate(str(corpus))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
 
    plt.imshow(wordcloud)
    plt.savefig(f'{bag_name}.png')
    # plt.show()
    return f"{bag_name}.png"


@app.get('/personality/wordbag', status_code=200)
async def get_personality_bag(bag: str):
    try:
        if os.path.exists(f'{bag}.png'):
            return FileResponse(f'{bag}.png', media_type="image/png", filename=f'{bag}.png')
        encoded_ds = pd.read_csv("mbti_cleaned.csv")
        encoded_ds = encoded_ds[encoded_ds["clean posts"].notna()] 
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        infj_df = encoded_ds[encoded_ds['type'] == bag]
        plot_image_path = await plot_wordcloud(infj_df["clean posts"], bag)
        if os.path.exists(plot_image_path):
            return FileResponse(plot_image_path, media_type="image/png", filename=plot_image_path)
    except Exception as e:
        return JSONResponse(
        status_code=404,
        content={"message": "Unable to generate personality word bag"},
    )


async def clean_data(post, all_stopwords, ps):
    post = re.sub(r"https?:\/\/(www)?.?([A-Za-z_0-9-]+)([\S])*", "", post) # Remove links
    post = re.sub("\|\|\|", "", post) # Remove |||
    post = re.sub("[0-9]", "", post) # Remove numbers
    post = re.sub("[^a-z]", " ", post) # Remove punctuation
    post = post.split()
    post = [word for word in post if word not in all_stopwords] # Stopwords removal
    post = " ". join([ps.stem(word) for word in post]) # Stemming
    return post

async def clean_pred_data(pred):
    random.shuffle((pred))

async def final_type(pred):
    await clean_pred_data(pred)
    mbti_types = [
                  ["E", "I"], 
                  ["N", "S"], 
                  ["F", "T"], 
                  ["J", "P"]
                ]
    ans = []
    for i, p in enumerate(pred):
        ans.append(mbti_types[i][p[0]])
    return "".join(ans)
    
@app.post('/personality/type', status_code=200)
async def get_personality_type(personality_response: dict):
    if not personality_response.get('answer'):
        return JSONResponse(
        status_code=400,
        content={"message": "Please provide correct answer for the questions"},
    )
    try:
        nltk.download("stopwords")
        all_stopwords = stopwords.words("english")
        all_stopwords.remove("not")
        ps = PorterStemmer()
        cv = pickle.load(open("cv.pkl", "rb"))
        xgb_model_introv = pickle.load(open(file_name_i, "rb"))
        xgb_model_sens = pickle.load(open(file_name_s, "rb"))
        xgb_model_think = pickle.load(open(file_name_t, "rb"))
        xgb_model_perc = pickle.load(open(file_name_p, "rb"))

        response = personality_response.get('answer').lower()
        response = await clean_data(response, all_stopwords, ps)
        response = cv.transform([response])
        prediction_q = [
              xgb_model_introv.predict(response),
              xgb_model_sens.predict(response),
              xgb_model_think.predict(response),
              xgb_model_perc.predict(response)
              ]
        personality_type = await final_type(prediction_q)
        return JSONResponse(
        status_code=200,
        content={"personality": f"{personality_type}"},
    )
    except Exception as e:
        return JSONResponse(
        status_code=404,
        content={"message": "Unable to predict, please try after sometime"},
    )
