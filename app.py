from flask import Flask, request, jsonify
from nltk.corpus import twitter_samples

app = Flask(__name__)

@app.route("/tweet", methods=["GET"])
def add_user():
    username = request.args['username']

    return jsonify(twitter_samples.fileids())


if __name__ == '__main__':
    app.run(debug=True)
