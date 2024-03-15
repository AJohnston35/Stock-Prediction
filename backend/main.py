from flask import Flask
app = Flask(__name__)
@app.route('/')
def hello_world():
    return "<p>Hello, World!</p>"

from markupsafe import escape
@app.route('/<name>')
def hello(name):
    return f'Hello, {escape(name)}!'

@app.route('/')
def index():
    return 'Index Page'

@app.route('/hello')
def hello():
    return 'Hello, World'

from flask import url_for

with app.test_request_context():
    print(url_for('index'))
    print(url_for('hello'))
    print(url_for('hello', name='John Doe'))
    
url_for('static', filename='style.css')