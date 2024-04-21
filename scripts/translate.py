import pandas as pd
import requests

path = "/"
df = pd.read_csv(path)
project_id = "/"
output_path = "/"
api_key = "/"


def clean_text(text):
    text = text.replace('&quot;', '"').replace('&#39;', "'").replace('&amp;', '&')
    text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
    return text


def translate_text(text, target_language, api_key):
    url = "https://translation.googleapis.com/language/translate/v2"
    params = {
        'key': api_key,
        'q': text,
        'target': target_language,
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        result = response.json()
        translated_text = result['data']['translations'][0]['translatedText']
        cleaned_text = clean_text(translated_text)
        return cleaned_text
    else:
        return "Translation Error"




def translated_to_csv(dataframe, name, key):
    articles = []
    highlights = []
    ids = []
    print(len(dataframe))
    
    for index, row in dataframe.iterrows():
        translated_article = translate_text(row["article"], "fi", key)
        translated_highlight = translate_text(row["highlights"], "fi", key)
        
        if "Translation Error" not in translated_article and "Translation Error" not in translated_highlight:
            ids.append(row["id"])
            articles.append(translated_article)
            highlights.append(translated_highlight)

    translated_df = pd.DataFrame({
        "id": ids,
        "artikkeli": articles,
        "tiivistys": highlights
    })

    print(f"Original DataFrame length: {len(dataframe)}, After filtering: {len(translated_df)}")
    translated_df.to_csv(name, index=False)





translated_to_csv(df, output_path, api_key)