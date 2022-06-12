import pickle,os
from flask import Flask,request,render_template
import pandas as pd
app= Flask(__name__)
model = pickle.load(open('xgb_model.pkl','rb'))

ll=0

@app.route('/')
def final():
    return render_template('index.html')

@app.route('/predict', methods = ['GET','POST'])
def predict():
    global ll
    if request.method=='POST':
        ind=request.form                  # returns an immutable dictionary
        
        l=dict(ind)
        in_str =[[float(l['online_order']), float(l['book_table']),float(l['votes']), float(l['location']),float(l['rest_type']),float(l['cuisine']), float(l['approx_cost']) ]]
        
        in_val = model.predict(pd.DataFrame(in_str))
        
        ll=in_val[0]
    
    return render_template('owner.html',result=float(ll)*10*2)



        
if __name__=='__main__':
    port = int(os.getenv("PORT"))
    app.run(host='0.0.0.0',port=port,debug=True)
