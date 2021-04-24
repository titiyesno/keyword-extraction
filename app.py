import flask
from flask_cors import CORS
from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField
from flask import render_template, redirect, flash, Response, jsonify
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import numpy as np
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
nltk.download('stopwords')
stopword = stopwords.words('english')
from keras.preprocessing.text import Tokenizer
import pickle
from wordcloud import WordCloud, STOPWORDS
import base64

app = flask.Flask(__name__)
app.config['SECRET_KEY'] = 'you-will-never-guess'
CORS(app)

model = None


class keyExtractForm(FlaskForm):
    transcription = TextAreaField('Transcription')
    submit = SubmitField('Extract Keyword')

@app.before_first_request
def load_keras():
    global model
    
    print("Loading model..")
    model = load_model('model.h5')
    
    print("Model loaded.")

@app.route("/predict", methods=["POST"])
def predict():
    show_extracted_keyword = False

    form = keyExtractForm()
    if form.validate_on_submit():
        # read the text
        text = form.transcription.data
        
        # load tokenizer
        with open('tokenizer.pickle', 'rb') as f:
            words_tokenizer = pickle.load(f)
        
        
        # preprocess data
        MAX_LEN = 485
        tokenizer = RegexpTokenizer(r'\w+')
        transcript_lower = text.lower()
        rem_punct = tokenizer.tokenize(transcript_lower)
        removing_stopwords = ' '.join([word for word in rem_punct if word not in stopword])
        global result
        result = ''.join([i for i in removing_stopwords if not i.isdigit()])
        word_index = words_tokenizer.word_index
        word_index['__PADDING__'] = 0
        X = words_tokenizer.texts_to_sequences([word_tokenize(result)])
        X = pad_sequences(X, padding = "post", truncating = "post", maxlen = MAX_LEN, value = 0)

        # predict
        X = np.asarray(X).astype('float32')
        test_output = model.predict(X)
        test_output = np.argmax(test_output, axis = -1)
        where_ = np.where(test_output[0] == 1)[0]
        output_keywords = np.take(X, where_)
        output_keywords = list(set(output_keywords))
        global keyword_output
        keyword_output = []
        for out in output_keywords:
            keyword_output.append(list(word_index.keys())[list(word_index.values()).index(out)])
        print(keyword_output)
        show_extracted_keyword = True

    return render_template('form.html', title='Keyword Extraction App', show_extracted_keyword=show_extracted_keyword, form=form, data=keyword_output)

@app.route("/wordcloud", methods=["GET"])
def generateWordCloud():
    # generate wordcloud
    wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, background_color='black', colormap='Pastel1', collocations=False, stopwords = STOPWORDS).generate(result)
    wordcloud.to_file("templates/wordcloud.png")
    with open("templates/wordcloud.png", "rb") as img_file:
        my_string = base64.b64encode(img_file.read()).decode('utf-8')
    return jsonify(path='data:image/png;base64,'+my_string)


@app.route("/download")
def exportToCsv():
    data = ','.join([i for i in keyword_output])
    return Response(
        data,
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename=keyword.csv"})

@app.route("/")
def renderForm():
    form = keyExtractForm()
    return render_template('form.html', title='Keyword Extraction App', form=form)

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_keras()
    app.run(debug=True)