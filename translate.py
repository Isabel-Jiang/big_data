from google_trans_new import google_translator
import pandas as pd

translator = google_translator(timeout=10)

reviews = pd.read_csv('./data/reviews_english.csv.gz')
pd.set_option('display.max_columns', None)

for i in range(20766):
    text_t = reviews['comments'].loc[i]

    try:
        translation = translator.translate(text_t)
        print(translation)
        reviews.loc[i, 'comments_english'] = translation
        print(i)
    except:
        reviews.loc[i, 'comments_english'] = text_t
        print('exception')
        continue

reviews.to_csv('./data/reviews_english.csv.gz', sep=',', header=True, index=False)
