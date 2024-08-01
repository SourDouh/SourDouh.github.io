#app.py
from flask import Flask, render_template, request

app = Flask(__name__)

def get_completion(prompt):
    return prompt.upper()
@app.route("/")
def home():    
    return render_template("index.html")
@app.route("/get")
def get_bot_response():    
    userText = request.args.get('msg')  
    response = get_completion(userText)  
    #return str(bot.get_response(userText)) 
    return response
if __name__ == "__main__":
    app.run(debug=True)
