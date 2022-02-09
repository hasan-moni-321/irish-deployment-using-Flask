import pickle
import numpy as np 
from crypt import methods
from flask import Flask, render_template, request

app = Flask(__name__) 

file_name = "models/logistic_model.sav"
loaded_model = pickle.load(open(file_name, 'rb'))

def prediction(s_l, s_w, p_l, p_w, loaded_model): 
    pre_data = np.array([s_l, s_w, p_l, p_w]) 
    pre_data_reshape = pre_data.reshape(1, -1) 
    pred_result = loaded_model.predict(pre_data_reshape)  
    return pred_result[0]
 

@app.route('/')
def input_data(): 
    return render_template('input.html') 


@app.route('/result', methods=["POST", "GET"]) 
def input(): 
    if request.method == "POST": 
        s_l = request.form['sepal_length']
        s_w = request.form['sepal_width']
        p_l = request.form['petal_length'] 
        p_w = request.form['petal_width'] 

        predicted_result = prediction(s_l, s_w, p_l, p_w, loaded_model)  

        return render_template('result.html', predicted = predicted_result) 

if __name__ == '__main__': 
    app.run(debug=True) 
