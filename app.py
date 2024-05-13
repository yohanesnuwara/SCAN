import nltk
#nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for generating PNG image
import io
import base64

from keras.models import load_model
model = load_model('model.h5')
import json
import random
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))

def merge_csv_data(csv1, csv2, sector):
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)
    merged_df = pd.merge(df1[df1['Sector'] == sector], df2[df2['Sector'] == sector], on='Sector')
    return merged_df.sort_values(by='Date', ascending=False)  # Get the last record

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def plot_timeseries(data):
  # Check if data is empty
  if data.empty:
    return "No data found for this sector."
  
  # Extract date and value columns
  dates = data['Date']
  valuesN, valuesP, valuesK = data['Value N'], data['Value P'], data['Value K']

  # Configure and display the plot
  plt.figure(figsize=(4.5,4))  # Adjust figure size as needed
  plt.plot(dates, valuesN, marker='o', linestyle='-', label='N')
  plt.plot(dates, valuesP, marker='o', linestyle='-', label='P')
  plt.plot(dates, valuesK, marker='o', linestyle='-', label='K')

  plt.xlabel('Date')
  plt.ylabel('NPK (kg/Ha)')
  plt.title(f'Historical NPK for Sector {data["Sector"].values[0]}')
  plt.grid(True)
  plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
  plt.tight_layout()
  plt.legend()
  
  # Instead of returning the plot, convert it to an image (optional)
  img_data = io.BytesIO()  # Use appropriate library (e.g., BytesIO)
  plt.savefig(img_data, format='png')
  img_data.seek(0)  # Rewind the buffer
  img_b64 = base64.b64encode(img_data.getvalue()).decode('utf-8')
  
  # Alternatively, display the plot directly (if using a graphical interface)
  plt.show()  # This line will block execution until the plot is closed
  return f"<img src='data:image/png;base64,{img_b64}' alt='Timeseries plot for Sector {data['Sector'].values[0]}'>"

def recommender(last_data):
    recommend_df = pd.read_csv("recommendation.csv")
    model_N = load_model("model_N.h5")
    model_P = load_model("model_P.h5")
    model_K = load_model("model_K.h5")
    value_N = last_data['Value N']
    value_P = last_data['Value P']
    value_K = last_data['Value K']
    age = last_data['Crop age'].values[0]
    if age=="0-2.5 year":
        age=[1]
    if age=="2.6-4 year":
        age=[2]
    if age=="4-10 year":
        age=[3]
    if age=="10-100 year":
        age=[4]

    phase = last_data['Phase'].values[0]
    if phase=="Vegetative Phase":
        phase=[1]
    if phase=="Flowering Phase":
        phase=[2]
    if phase=="Fruiting Phase":
        phase=[3]
    if phase=="Riping Phase":
        phase=[4]

    # Classify status of NPK (very low/low/enough/too much)
    df_N = pd.DataFrame({'Crop age': age, 'Phase': phase, 'Value N': value_N})
    df_P = pd.DataFrame({'Crop age': age, 'Phase': phase, 'Value N': value_P})
    df_K = pd.DataFrame({'Crop age': age, 'Phase': phase, 'Value N': value_K})

    pred_class_N = np.argmax(model_N.predict(df_N), axis=1)[0]
    print(recommend_df["Class"])
    treatment_N = recommend_df[recommend_df["Class"]==pred_class_N]["Recommendation_N"].values[0]
    if pred_class_N==1:
        pred_class_N="Very low"
    if pred_class_N==2:
        pred_class_N="Low"
    if pred_class_N==3:
        pred_class_N="Enough"
    if pred_class_N==4:
        pred_class_N="Too much"          

    pred_class_P = np.argmax(model_P.predict(df_P), axis=1)[0]
    treatment_P = recommend_df[recommend_df["Class"]==pred_class_P]["Recommendation_P"].values[0]
    if pred_class_P==1:
        pred_class_P="Very low"
    if pred_class_P==2:
        pred_class_P="Low"
    if pred_class_P==3:
        pred_class_P="Enough"
    if pred_class_P==4:
        pred_class_P="Too much"      
    
    pred_class_K = np.argmax(model_K.predict(df_K), axis=1)[0]
    treatment_K = recommend_df[recommend_df["Class"]==pred_class_K]["Recommendation_K"].values[0]
    if pred_class_K==1:
        pred_class_K="Very low"
    if pred_class_K==2:
        pred_class_K="Low"
    if pred_class_K==3:
        pred_class_K="Enough"
    if pred_class_K==4:
        pred_class_K="Too much"  

    return pred_class_N, pred_class_P, pred_class_K, treatment_N, treatment_P, treatment_K

def Sequential_Input_LSTM(df, input_sequence=7):
    df_np = df.to_numpy()
    X = []
    y = []
    
    for i in range(len(df_np) - input_sequence):
        row = [a for a in df_np[i:i + input_sequence]]
        X.append(row)
        label = df_np[i + input_sequence]
        y.append(label)
        
    return np.array(X), np.array(y)

def futureForecast(df, col, n_input, n_features, forecast_timeperiod, model):

    x_input = np.array(df[len(df)-n_input:])

    temp_input=list(x_input)

    lst_output=[]
    i=0

    while(i < forecast_timeperiod):

        if(len(temp_input) > n_input):

            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape((1, n_input, n_features))
            yhat = model.predict(x_input, verbose=0)
            temp_input.append(yhat[0][0])
            temp_input = temp_input[1:]
            lst_output.append(yhat[0][0])

            i=i+1

        else:
            x_input = x_input.reshape((1, n_input, n_features))
            yhat = model.predict(x_input, verbose=0)
            #print(yhat[0])
            temp_input.append(yhat[0][0])
            lst_output.append(yhat[0][0])

            i=i+1
            
    return lst_output

def inverse_transform(y_pred, mean=56.437, std=4.431):
    return np.array(y_pred)*std + mean

def forecast_weather(column='outdoor_humidity', mean=56.437, std=4.431):
    weather_data = "D:\\Agrari Chatbot\\keras\\humidity_data.csv"
    model_path = "D:\\Agrari Chatbot\\keras\\model_humidity.h5"
    weather_df = pd.read_csv(weather_data)

    df_for_test = weather_df[column]
    df_for_test_normalized = (df_for_test - mean) / std
    # X_blind_test, y_blind_test = Sequential_Input_LSTM(df_for_test_normalized, input_sequence=7)

    # Forecast next 3 days
    model = load_model(model_path)
    y_3days_norm = futureForecast(df_for_test_normalized, 'outdoor_humidity', 7, 1, 3, model)

    # Inverse transform
    y_3days = inverse_transform(y_3days_norm)    
    return y_3days

def forecast_market(column='price', mean=30281.53, std=1029.09):
    weather_data = "D:\\Agrari Chatbot\\keras\\commodity_price.csv"
    model_path = "D:\\Agrari Chatbot\\keras\\model_price.h5"
    weather_df = pd.read_csv(weather_data)

    df_for_test = weather_df[column]
    df_for_test_normalized = (df_for_test - mean) / std
    # X_blind_test, y_blind_test = Sequential_Input_LSTM(df_for_test_normalized, input_sequence=7)

    # Forecast next 3 days
    model = load_model(model_path)
    y_3days_norm = futureForecast(df_for_test_normalized, 'outdoor_humidity', 7, 1, 3, model)

    # Inverse transform
    y_3days = inverse_transform(y_3days_norm, mean=30281.53, std=1029.09)    
    return y_3days

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    print(ints)
    print(res)
    csv1 = "D:\\Agrari Chatbot\\sector.csv"
    csv2 = "D:\\Agrari Chatbot\\sensor.csv"
    if res == "Loading current data for Sector A":
        # Call the merge function with sector A
        data = merge_csv_data(csv1, csv2, "A")
        data = data.head(1)
        # Check if data is empty
        if data.empty:
            return "No data found for Sector A."
        else:
            # Access the desired value (e.g., value N) from the DataFrame
            return f"Current data for Sector A per {data['Date'].values[0]}: N = {data['Value N'].values[0]} kg/Ha, P = {data['Value P'].values[0]} kg/Ha, K = {data['Value K'].values[0]} kg/Ha"
            
    elif res == "Loading current data for Sector B":
        # Similar logic for Sector B
        data = merge_csv_data(csv1, csv2, "B")
        data = data.head(1)
        if data.empty:
            return "No data found for Sector B."
        else:
            return f"Current data for Sector B per {data['Date'].values[0]}: N = {data['Value N'].values[0]} kg/Ha, P = {data['Value P'].values[0]} kg/Ha, K = {data['Value K'].values[0]} kg/Ha"
    elif res == "Checking historical data for Sector A":
        data = merge_csv_data(csv1, csv2, "A")
        return plot_timeseries(data)
    elif res == "Checking historical data for Sector B":
        data = merge_csv_data(csv1, csv2, "B")    
        return plot_timeseries(data)
    elif res == "Loading NPK status and recommend for sector A":
        data = merge_csv_data(csv1, csv2, "A")      
        data = data.head(1)
        class_N, class_P, class_K, treatment_N, treatment_P, treatment_K = recommender(data)
        return f"Status N {class_N} P {class_P} K {class_K}. \n Treatments needed for N: {treatment_N}, for P: {treatment_P}, for K: {treatment_K}"  
    elif res == "Loading NPK status and recommend for sector B":
        data = merge_csv_data(csv1, csv2, "B")      
        data = data.head(1)
        class_N, class_P, class_K, treatment_N, treatment_P, treatment_K = recommender(data)
        return f"Status N {class_N} P {class_P} K {class_K}. \n Treatments needed for N: {treatment_N}, for P: {treatment_P}, for K: {treatment_K}"      
    elif res == "Generating weather forecast":
        forecast = forecast_weather()
        return f"The humidity for the next 3 days are {forecast[0]:.2f}%, {forecast[1]:.2f}%, and {forecast[2]:.2f}%"
    elif res == "Generating market price forecast":
        forecast = forecast_market()
        return f"The cacao producer price for the next 3 days are Rp {int(forecast[0])},-, Rp {int(forecast[1])},-, and Rp {int(forecast[2])},-"

    return res


from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)


if __name__ == "__main__":
    app.run()