import datetime
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from flask import Flask, render_template, send_file, request, redirect, url_for, Response, jsonify
from flask_sqlalchemy import SQLAlchemy
import json
import io
from io import BytesIO
import tweepy
import random
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import json
from ibm_watson import ToneAnalyzerV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


x = datetime.datetime.now()




##########################################################################
################################  flask setup ############################
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///info.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)








class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    firstname = db.Column(db.String(80), unique=True, nullable=False)
    lastname = db.Column(db.String(80), unique=True, nullable=False)
    emailid = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(80), unique=True, nullable=False)
    repeat_password = db.Column(db.String(80), unique=True, nullable=False)

    def __repr__(self):
        return '<User %r' % self.emailid


class Post(db.Model):
    
    __tablename__ = 'post'

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(40), nullable=False,default=x.strftime("%m-%d"))
    positive = db.Column(db.Integer, nullable=False)
    negative = db.Column(db.Integer, nullable=False)
    neutral = db.Column(db.Integer, nullable=False)

    def __init__(self, positive,negative,netural):
        self.positive = positive
        self.negative = negative
        self.neutral = netural


authenticator = IAMAuthenticator(
    'ry1CMXoffu8dVsUrlv3n6i79Pq6B1AEzE3MmfvpEKZU3')
tone_analyzer = ToneAnalyzerV3(
    version='2017-09-21',
    authenticator=authenticator
)
tone_analyzer.set_service_url('https://api.eu-gb.tone-analyzer.watson.cloud.ibm.com/instances/1a4eab60-917b-4fef-a93a-30b08d3555b8')





###### Load Twitter credentials ####
log = pd.read_csv('login.csv')
consumerKey = log['Key'][0]
consumerSecret = log['Key'][1]
accessToken = log['Key'][2]
accessTokenScret = log['Key'][3]

authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret)
#set the access token
authenticate.set_access_token(accessToken, accessTokenScret)
api = tweepy.API(authenticate, wait_on_rate_limit=True)

pmposts = api.user_timeline(screen_name="narendramodi", count=100, lang=["en"], tweet_mode="extended")
df = pd.DataFrame([tweet.full_text for tweet in pmposts], columns=['PmTweets'])

#HRD ministry tweets
hrdposts = api.user_timeline(screen_name="HRDMinistry", count=100, lang="en", tweet_mode="extended")
df['HrdTweets'] = pd.DataFrame([tweet.full_text for tweet in hrdposts], columns=['HrdTweets'])

#Finance Tweets
finposts = api.user_timeline(screen_name="FinMinIndia", count=100, lang="en", tweet_mode="extended")
df['FinTweets'] = pd.DataFrame([tweet.full_text for tweet in finposts], columns=['FinTweets'])

search_results = api.search(q="#COVID-19", geocode='28.644800,77.216721,3000km', count=10000, lang="en", result_type="recent")
covid = pd.DataFrame([tweet.text for tweet in search_results], columns=['Covid Tweets'])


bengaluru_results = api.search(q="#Bengaluru covid", geocode='12.972442,77.580643,100km', count=1000, lang="en", result_type="recent")
covid['BengaluruTweets'] = pd.DataFrame([tweet.text for tweet in bengaluru_results], columns=['BengaluruTweets'])


chennai_results = api.search(q="#covid", geocode='13.067439,80.237617,200km', count=1000, lang="en", result_type="recent")
covid['CTweets'] = pd.DataFrame([tweet.text for tweet in chennai_results], columns=['CTweets'])

mumbai_results = api.search(q="#Mumbai #COVID", geocode='19.076090,72.877426,100km',count=1000, lang="en", result_type="recent")
covid['MuTweets'] = pd.DataFrame([tweet.text for tweet in mumbai_results], columns=['MuTweets'])


delhi_results = api.search(q="#COVID", geocode='28.644800,77.216721,150km',count=1000, lang="en", result_type="recent")
covid['DTweets'] = pd.DataFrame([tweet.text for tweet in delhi_results], columns=['DTweets'])




@app.route('/dashboard')
def index():
    df['PmTweets'] = df['PmTweets'].apply(cleanTxt)
    df['HrdTweets'] = df['HrdTweets'].apply(cleanTxt)

    df['FinTweets'] = df['FinTweets'].apply(cleanTxt)

    covid['Covid Tweets'] = covid['Covid Tweets'].apply(cleanTxt)

    covid['BengaluruTweets'] = covid['BengaluruTweets'].apply(cleanTxt)

    covid['CTweets'] = covid['CTweets'].apply(cleanTxt)

    covid['MuTweets'] = covid['MuTweets'].apply(cleanTxt)

    covid['DTweets'] = covid['DTweets'].apply(cleanTxt)

    df['PmSubjectivity'] = df['PmTweets'].apply(getSubjectivity)
    df['PmPolarity'] = df['PmTweets'].apply(getPolarity)
    df['PmAnalysis'] = df['PmPolarity'].apply(getAnalysis)

    ###################### HRD NLP Prossig ##################
    df['HrdSubjectivity'] = df['HrdTweets'].apply(getSubjectivity)
    df['HrdPolarity'] = df['HrdTweets'].apply(getPolarity)
    df['HrdAnalysis'] = df['HrdPolarity'].apply(getAnalysis)

    ################# Finance  NLP Prossig ##################
    df['FinSubjectivity'] = df['FinTweets'].apply(getSubjectivity)
    df['FinPolarity'] = df['FinTweets'].apply(getPolarity)
    df['FinAnalysis'] = df['FinPolarity'].apply(getAnalysis)

    ####################covid 19 NLP Prossig ##################
    covid['Subjectivity'] = covid['Covid Tweets'].apply(getSubjectivity)
    covid['Polarity'] = covid['Covid Tweets'].apply(getPolarity)
    covid['Analysis'] = covid['Polarity'].apply(getAnalysis)

    ##############################################################################################

    covid['BengaluruSubjectivity'] = covid['BengaluruTweets'].apply(getSubjectivity)
    covid['BengaluruPolarity'] = covid['BengaluruTweets'].apply(getPolarity)
    covid['BangaluruAnalysis'] = covid['BengaluruPolarity'].apply(getAnalysis)

    covid['ChennaiSubjectivity'] = covid['CTweets'].apply(getSubjectivity)
    covid['ChennaiPolarity'] = covid['CTweets'].apply(getPolarity)
    covid['ChennaiAnalysis'] = covid['ChennaiPolarity'].apply(getAnalysis)


    covid['MumbaiSubjectivity'] = covid['MuTweets'].apply(getSubjectivity)
    covid['MumbaiPolarity'] = covid['MuTweets'].apply(getPolarity)
    covid['MumbaiAnalysis'] = covid['MumbaiPolarity'].apply(getAnalysis)

    covid['DelhiSubjectivity'] = covid['DTweets'].apply(getSubjectivity)
    covid['DelhiPolarity'] = covid['DTweets'].apply(getPolarity)
    covid['DelhiAnalysis'] = covid['DelhiPolarity'].apply(getAnalysis)

    ######################################################################################################

    ################################# calculate city sentiment percentage ###############################
    bangpostcount = covid['BengaluruPolarity'].count()

    sortedCOVIDB = covid.sort_values(by=['BengaluruPolarity'])
    counts = 0

    for i in range(0, sortedCOVIDB.shape[0]):
        if(sortedCOVIDB['BangaluruAnalysis'][i] == 'positive'):
            counts = counts + 1

    if counts != 0:
        bangpercentage = percentage(bangpostcount, counts)
    ######################################################################################################

    ######################################################################################################

    chepostcount = covid['ChennaiPolarity'].count()

    sortedCOVIDC = covid.sort_values(by=['ChennaiPolarity'])
    countC = 0

    for i in range(0, sortedCOVIDC.shape[0]):
        if(sortedCOVIDC['ChennaiAnalysis'][i] == 'positive'):
            countC = countC + 1

    if countC != 0:
        chepercentage = percentage(chepostcount, countC)




    #######################################################################################################

    ######################################################################################################

    delpostcount = covid['DelhiPolarity'].count()

    sortedCOVIDD = covid.sort_values(by=['DelhiPolarity'])
    countD = 0

    for i in range(0, sortedCOVIDD.shape[0]):
        if(sortedCOVIDD['DelhiAnalysis'][i] == 'positive'):
            countD = countD + 1

    if countD != 0:
        delpercentage = percentage(delpostcount, countD)

    #######################################################################################################


    ######################################################################################################

    mupostcount = covid['MumbaiPolarity'].count()

    sortedCOVIDM = covid.sort_values(by=['MumbaiPolarity'])
    countM = 0

    for i in range(0, sortedCOVIDM.shape[0]):
        if(sortedCOVIDM['MumbaiAnalysis'][i] == 'positive'):
            countM = countM + 1

    if countM != 0:
        mupercentage = percentage(mupostcount, countM)

    #######################################################################################################

    ########################## Count no of +ve PM tweets  ###################################
    pmpostcount = df['PmPolarity'].count()

    sortedDF2 = df.sort_values(by=['PmPolarity'])
    count = 0
    for i in range(0, sortedDF2.shape[0]):
        if(sortedDF2['PmAnalysis'][i] == 'positive'):
            count = count + 1

    if count != 0:
        govpercentage = percentage(pmpostcount, count)
    ####################################################################################
    hrdpostcount = df['HrdPolarity'].count()

    sortedDF3 = df.sort_values(by=['HrdPolarity'])
    count = 0
    for i in range(0, sortedDF3.shape[0]):
        if(sortedDF3['HrdAnalysis'][i] == 'positive'):
            count = count + 1

    if count != 0:
        Hrdpercentage = percentage(hrdpostcount, count)
########################## Count no of +ve Finance  tweets  ###################################
    finpostcount = df['FinPolarity'].count()

    sortedDF4 = df.sort_values(by=['FinPolarity'])
    count = 0
    for i in range(0, sortedDF4.shape[0]):
        if(sortedDF4['FinAnalysis'][i] == 'positive'):
            count = count + 1

    if count != 0:
        Finpercentage = percentage(finpostcount, count)
########################## Count no of  COVID tweets  ###################################
    covidpostcount = covid['Polarity'].count()

    sortedCOVID = covid.sort_values(by=['Polarity'])
    count = 0
    negative = 0
    netural = 0

    for i in range(0, sortedCOVID.shape[0]):
        if(sortedCOVID['Analysis'][i] == 'positive'):
            count = count + 1

        if(sortedCOVID['Analysis'][i] == 'Negative'):
            negative = negative + 1

    if count != 0:
        covidpercentage = percentage(covidpostcount, count)
        negativeper = percentage(covidpostcount, negative)
        netural = 100 - (covidpercentage+negativeper)
        currentdate = x.strftime("%m-%d")
        dates = Post.query.filter_by(date=currentdate).first()
        
        if Post.query.filter_by(date=currentdate).first() and dates.positive < count:
            postcount = Post(positive=count, negative=negative, netural=netural)
            db.session.add(postcount)
            db.session.commit()

        elif currentdate != dates:
            postcount = Post(positive=count, negative=negative, netural=netural)
            db.session.add(postcount)
            db.session.commit()
        
    

    postid = Post.query.all()

    datess = []
    positivess = []
    negativess = []
    neutralss = []
    i = 0
    for post in postid:
        datess.append(post.date)
        positivess.append(post.positive)
        negativess.append(post.negative)
        neutralss.append(post.neutral)
        i = +i
    res = str(datess)[1:-1]


    ####################################################################################################
    ######################################### WORD CLOUD ###############################################

    allwords = ' '.join([twts for twts in covid['Covid Tweets']])

    wordCloud = WordCloud(width=800, height=500, random_state=36,
                          max_font_size=119).generate(allwords)
    plt.figure()
    plt.imshow(wordCloud, interpolation="bilinear")
    plt.axis("off")
    wordCloud.to_file("static/img/irst_review.png")

    return render_template('index.html', govpercentage=govpercentage, Hrdpercentage=Hrdpercentage, Finpercentage=Finpercentage, covidpercentage=covidpercentage, negative=negativeper, netural=netural, bangpercentage=bangpercentage, chepercentage=chepercentage, mupercentage=mupercentage, delpercentage=delpercentage, res=res, positivess=positivess, negativess=negativess, neutralss=neutralss)

################################################################################


###############################################################################
########################### find percentage ##################################
def percentage(pmpostcount, count):
    pers = (count/pmpostcount)*100
    per = int(round(pers))
    return per

#################################################################################
@app.route('/chart')
def chart():
    return render_template('charts.html')


@app.route('/data')
def data():
    df['PmTweets'] = df['PmTweets'].apply(cleanTxt)
    df['PmSubjectivity'] = df['PmTweets'].apply(getSubjectivity)
    df['PmPolarity'] = df['PmTweets'].apply(getPolarity)
    df['PmAnalysis'] = df['PmPolarity'].apply(getAnalysis)

    fig, ax = plt.subplots(figsize=(18, 17))
    df['PmAnalysis'].value_counts()

    plt.title('PM Sentiment Analysis')
    plt.xlabel('Sentiment')
    plt.ylabel('Counts')
    df['PmAnalysis'].value_counts().plot(kind='bar', color=(0.2, 0.4, 0.6, 0.6))
    canvas = FigureCanvas(fig)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')


@app.route('/hrd')
def hrd():
    df['HrdTweets'] = df['HrdTweets'].apply(cleanTxt)
    df['HrdSubjectivity'] = df['HrdTweets'].apply(getSubjectivity)
    df['HrdPolarity'] = df['HrdTweets'].apply(getPolarity)
    df['HrdAnalysis'] = df['HrdPolarity'].apply(getAnalysis)

    fig, ax = plt.subplots(figsize=(18, 17))
    df['HrdAnalysis'].value_counts()

    plt.title('HRD Sentiment Analysis')
    plt.xlabel('Sentiment')
    plt.ylabel('Counts')
    df['HrdAnalysis'].value_counts().plot(kind='bar')
    canvas = FigureCanvas(fig)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')


@app.route('/fin')
def fin():
    df['FinTweets'] = df['FinTweets'].apply(cleanTxt)
    df['FinSubjectivity'] = df['FinTweets'].apply(getSubjectivity)
    df['FinPolarity'] = df['FinTweets'].apply(getPolarity)
    df['FinAnalysis'] = df['FinPolarity'].apply(getAnalysis)

    fig, ax = plt.subplots(figsize=(18, 17))
    df['FinAnalysis'].value_counts()

    plt.title('Finance Sentiment Analysis')
    plt.xlabel('Sentiment')
    plt.ylabel('Counts')
    df['FinAnalysis'].value_counts().plot(kind='bar')
    canvas = FigureCanvas(fig)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')


@app.route('/corona')
def corona():
    covid['Covid Tweets'] = covid['Covid Tweets'].apply(cleanTxt)
    covid['Subjectivity'] = covid['Covid Tweets'].apply(getSubjectivity)
    covid['Polarity'] = covid['Covid Tweets'].apply(getPolarity)
    covid['Analysis'] = covid['Polarity'].apply(getAnalysis)

    fig, ax = plt.subplots(figsize=(18, 17))
    covid['Analysis'].value_counts()

    plt.title('Covid Sentiment Analysis')
    plt.xlabel('Sentiment')
    plt.ylabel('Counts')
    covid['Analysis'].value_counts().plot(kind='bar')
    canvas = FigureCanvas(fig)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')


###############################################################################################
################################ Cities Sentiment Bar ########################################

@app.route('/beng')
def beng():
    covid['BengaluruTweets'] = covid['BengaluruTweets'].apply(cleanTxt)
    covid['BengaluruSubjectivity'] = covid['BengaluruTweets'].apply(getSubjectivity)
    covid['BengaluruPolarity'] = covid['BengaluruTweets'].apply(getPolarity)
    covid['BangaluruAnalysis'] = covid['BengaluruPolarity'].apply(getAnalysis)

    fig, ax = plt.subplots(figsize=(18, 17))
    covid['BangaluruAnalysis'].value_counts()

    plt.title('Bebgaluru Sentiment Analysis')
    plt.xlabel('Sentiment')
    plt.ylabel('Counts')
    covid['BangaluruAnalysis'].value_counts().plot(kind='bar')
    canvas = FigureCanvas(fig)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')


@app.route('/che')
def che():
    covid['CTweets'] = covid['CTweets'].apply(cleanTxt)
    covid['ChennaiSubjectivity'] = covid['CTweets'].apply(getSubjectivity)
    covid['ChennaiPolarity'] = covid['CTweets'].apply(getPolarity)
    covid['ChennaiAnalysis'] = covid['ChennaiPolarity'].apply(getAnalysis)

    fig, ax = plt.subplots(figsize=(18, 17))
    covid['ChennaiAnalysis'].value_counts()

    plt.title('Chennai Sentiment Analysis')
    plt.xlabel('Sentiment')
    plt.ylabel('Counts')
    covid['ChennaiAnalysis'].value_counts().plot(kind='bar')
    canvas = FigureCanvas(fig)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')


@app.route('/mum')
def mum():
    covid['MuTweets'] = covid['MuTweets'].apply(cleanTxt)
    covid['MumbaiSubjectivity'] = covid['MuTweets'].apply(getSubjectivity)
    covid['MumbaiPolarity'] = covid['MuTweets'].apply(getPolarity)
    covid['MumbaiAnalysis'] = covid['MumbaiPolarity'].apply(getAnalysis)

    fig, ax = plt.subplots(figsize=(18, 17))
    covid['MumbaiAnalysis'].value_counts()

    plt.title('Mumbai Sentiment Analysis')
    plt.xlabel('Sentiment')
    plt.ylabel('Counts')
    covid['MumbaiAnalysis'].value_counts().plot(kind='bar')
    canvas = FigureCanvas(fig)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')


@app.route('/delhi')
def delhi():
    covid['DeTweets'] = covid['DeTweets'].apply(cleanTxt)
    covid['DelhiSubjectivity'] = covid['DeTweets'].apply(getSubjectivity)
    covid['DelhiPolarity'] = covid['DeTweets'].apply(getPolarity)
    covid['DelhiAnalysis'] = covid['DelhiPolarity'].apply(getAnalysis)

    fig, ax = plt.subplots(figsize=(18, 17))
    covid['DelhiAnalysis'].value_counts()

    plt.title('Delhi Sentiment Analysis')
    plt.xlabel('Sentiment')
    plt.ylabel('Counts')
    covid['DelhiAnalysis'].value_counts().plot(kind='bar')
    canvas = FigureCanvas(fig)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')




##################################################################################
##################################################################################


##################################################################################
######################## Text Preprocess ########################################
def cleanTxt(text):
    text = re.sub(r'@[A-Z a-z 0-9]+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r'https?:\/\/\S+', '', text)
    
    return text


###################################################################################


def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity


def getPolarity(text):
  return TextBlob(text).sentiment.polarity
#########################################################################################


def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'positive'

########################################################################################


@app.route('/test', methods=['GET', 'POST'])
def test():
    sentiment = ()
    analyasis = ()
    if request.method == 'POST':
        
        text = request.form['text']

        score = TextBlob(text).sentiment.polarity
        analyasis = getAnalysis(score)
        tone_analysis = tone_analyzer.tone({'text': text}, content_type='application/json').get_result()
        tonss = tone_analysis['document_tone']['tones']
        
        for i in tonss:
            sentiment = i['tone_id']
    
    return render_template('graph.html', sentiment=sentiment, analyasis=analyasis)









################################### login ##############################################
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        with open('data.json', 'r') as filejson:
            data = json.load(filejson)
            if data['email'] == email:
                if data['password'] == password:
                    return redirect(url_for('index'))
            else:
                return redirect(url_for('login'))
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        fname = request.form['fname']
        lname = request.form['lname']
        email = request.form['email']
        psw = request.form['psw']
        rpsw = request.form['rpsw']
        if psw == rpsw:
            data = {'firstname': fname, 'lastname': lname,
                    'email': email, 'password': psw, 'repeatpassword': rpsw}
            with open('data.json', 'w') as filejson:
                json.dump(data, filejson)
            useradd = User(firstname=fname, lastname=lname,
                           emailid=email, password=psw, repeat_password=rpsw)
            db.session.add(useradd)
            db.session.commit()
            return redirect(url_for('login'))
        else:
            return render_template('register.html')
    return render_template('register.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)






