from flask import Flask,render_template,request
import pandas as pd
from sklearn import preprocessing
import pickle

app=Flask(__name__)
df = pd.read_csv('crop_production.csv', encoding='utf-8')
df = df[df['State_Name'] == "Andhra Pradesh"]
df = df[df['Crop_Year'] >= 2004]
x = df[['Area']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(x)
@app.route("/")
def home():
    return render_template("Userpage.html")

@app.route("/predict",methods=["GET","POST"])
def predict():
    district=request.form['district']
    season=request.form['season']
    cropname=request.form['cropname']
    area=request.form['area']
    area=min_max_scaler.transform([[area]])

    ndict = {'Area': 0.00}
    ndf = pd.DataFrame([ndict])
    ndf['ANANTAPUR'] = 0
    ndf['CHITTOOR'] = 0
    ndf['EAST GODAVARI'] = 0
    ndf['GUNTUR'] = 0
    ndf['KADAPA'] = 0
    ndf['KRISHNA'] = 0
    ndf['KURNOOL'] = 0
    ndf['PRAKASAM'] = 0
    ndf['SPSR NELLORE'] = 0
    ndf['SRIKAKULAM'] = 0
    ndf['VISAKHAPATNAM'] = 0
    ndf['VIZIAYANAGARAM'] = 0
    ndf['WEST GODAVARI'] = 0

    ndf['Kharif'] = 0
    ndf['Rabi'] = 0
    ndf['Whole Year'] = 0

    ndf['Arecanut'] = 0
    ndf['Arhar/Tur'] = 0
    ndf['Bajra'] = 0
    ndf['Banana'] = 0
    ndf['Brinjal'] = 0
    ndf['Cabbage'] = 0
    ndf['Cashewnut'] = 0
    ndf['Castor seed'] = 0
    ndf['Coconut'] = 0
    ndf['Coriander'] = 0
    ndf['Cotton(lint)'] = 0
    ndf['Cowpea(Lobia)'] = 0
    ndf['Dry chillies'] = 0
    ndf['Dry ginger'] = 0
    ndf['Garlic'] = 0
    ndf['Ginger'] = 0
    ndf['Gram'] = 0
    ndf['Grapes'] = 0
    ndf['Groundnut'] = 0
    ndf['Horse-gram'] = 0
    ndf['Jowar'] = 0
    ndf['Lemon'] = 0
    ndf['Linseed'] = 0
    ndf['Maize'] = 0
    ndf['mango'] = 0
    ndf['Mesta'] = 0
    ndf['Moong(Green Gram)'] = 0
    ndf['Niger seed'] = 0
    ndf['Onion'] = 0
    ndf['Other  Rabi pulses'] = 0
    ndf['Other Kharif pulses'] = 0
    ndf['Papaya'] = 0
    ndf['Pome Granet'] = 0
    ndf['Potato'] = 0
    ndf['Ragi'] = 0
    ndf['Rapeseed &Mustard'] = 0
    ndf['Rice'] = 0
    ndf['Safflower'] = 0
    ndf['Sannhamp'] = 0
    ndf['Sapota'] = 0
    ndf['Sesamum'] = 0
    ndf['Small millets'] = 0
    ndf['Soyabean'] = 0
    ndf['Sugarcane'] = 0
    ndf['Sunflower'] = 0
    ndf['Sweet potato'] = 0
    ndf['Tapioca'] = 0
    ndf['Tobacco'] = 0
    ndf['Tomato'] = 0
    ndf['Turmeric'] = 0
    ndf['Urad'] = 0
    ndf['Wheat'] = 0
    ndf['other oilseeds'] = 0

    '''ndf['2004'] = 0
    ndf['2005'] = 0
    ndf['2006'] = 0
    ndf['2007'] = 0
    ndf['2008'] = 0
    ndf['2009'] = 0
    ndf['2010'] = 0
    ndf['2011'] = 0
    ndf['2012'] = 0
    ndf['2013'] = 0
    ndf['2014'] = 1'''

    ndf['Andhra Pradesh'] = 1
    ndf["Area"]=area
    if ((season not in ndf) or (district not in ndf) or (cropname not in ndf)):
        return render_template("error.html")
    ndf[district] = 1
    ndf[season] = 1
    ndf[cropname] = 1
    # ndf[year]=1

    Pkl_Filename = "Pickle_RL_Model.pkl"
    with open(Pkl_Filename, 'rb') as file:
        gb = pickle.load(file)
    pred=gb.predict(ndf)
    return render_template("result.html",result=pred)



if __name__=="__main__":
    app.run(debug=True)
