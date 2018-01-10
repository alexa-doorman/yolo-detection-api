import io
import json
import base64
import logging
from datetime import datetime
import os

from flask import Flask, jsonify, request, make_response
from flask_redis import FlaskRedis
from flask_httpauth import HTTPBasicAuth
from redis import StrictRedis

from werkzeug.utils import secure_filename
import cv2
import numpy as np

import load_model

log_formatter = logging.Formatter(
    "%(asctime)s [ %(threadName)-12.12s ] [ %(levelname)-5.5s ]  %(message)s")
logger = logging.getLogger()

file_handler = logging.FileHandler("info.log")
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

logger.setLevel(logging.INFO)


auth = HTTPBasicAuth()


USER_DATA = {
    os.environ['DETECT_API_USERNAME']: os.environ['DETECT_API_PASSWORD']
}


@auth.verify_password
def verify(username, password):
    if not (username and password):
        return False
    return USER_DATA.get(username) == password


class DecodedRedis(StrictRedis):
    @classmethod
    def from_url(cls, url, db=None, **kwargs):
        kwargs['decode_responses'] = True
        return StrictRedis.from_url(url, db, **kwargs)


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = os.environ.get(
    'MAX_CONTENT_LENGTH', 5 * 1024 * 1024)  # max 5 MB
app.config['REDIS_URL'] = os.environ.get(
    'REDIS_URL', 'redis://localhost:6379/0')
REDIS_STORE = FlaskRedis.from_custom_provider(DecodedRedis, app)
REDIS_STORE.init_app(app)
THROTTLE = int(os.environ.get('THROTTLE_SECONDS', 5))

ALLOWED_EXTENSIONS = ['jpeg', 'jpg']


def allowed_file(filename):
    # http://flask.pocoo.org/docs/0.12/patterns/fileuploads/
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _make_error(status_code, message):
    return make_response(jsonify({'status': 'error', 'message': message}), status_code)


def make_400(message):
    return _make_error(401, message)


def make_429(message):
    return _make_error(429, message)


def make_404(message):
    return _make_error(404, message)


@app.before_first_request
def init_once():
    global tfnet
    tfnet = load_model.pb_yolo()


USER_METADATA_VERSION = 2
DEFAULT_EXPIRE_SECONDS = 3600


def make_ip_config():
    return {
        'now': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'counter': 1,
        'version': USER_METADATA_VERSION,
        'requests': 1
    }


@app.before_request
def before_request():
    ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)

    user_metadata = REDIS_STORE.get(ip)
    if user_metadata is None:
        REDIS_STORE.set(ip, json.dumps(make_ip_config()),
                        ex=DEFAULT_EXPIRE_SECONDS)
    else:
        user_metadata = json.loads(user_metadata)
        if user_metadata['version'] != USER_METADATA_VERSION:
            user_metadata = make_ip_config()

        elapsed = (datetime.now() -
                   datetime.strptime(user_metadata['now'], '%Y-%m-%d %H:%M:%S'))

        user_metadata['requests'] += 1

        if app.debug or elapsed.total_seconds() > THROTTLE:
            user_metadata['counter'] += 1
            user_metadata['now'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            REDIS_STORE.set(ip, json.dumps(user_metadata),
                            ex=DEFAULT_EXPIRE_SECONDS)
        else:
            REDIS_STORE.set(ip, json.dumps(user_metadata),
                            ex=DEFAULT_EXPIRE_SECONDS)
            return make_429("Please wait before requesting again")


@app.route('/')
@auth.login_required
def index():
    return jsonify({'message': 'the model was loaded successfully :D'})


@app.route('/detect', methods=['GET', 'POST'])
@auth.login_required
def detect():
    if request.method == 'GET':
        return make_404('GET not permitted')
    else:
        if 'b64image' in request.form:
            logger.info('found b64 image with len {0}'.format(
                len(request.form['b64image'])))
            b64_img = request.form['b64image']
            try:
                img_decoded = base64.b64decode(b64_img)
                jpg = np.frombuffer(img_decoded, dtype=np.uint8)
                img = cv2.imdecode(jpg, flags=1)
            except Exception as e:
                logger.error(e)
                return make_404('Decode error with byte image')
            results = tfnet.return_predict(img)
            if results:
                for result in results:
                    for k, v in result.items():
                        if isinstance(v, np.generic):
                            result[k] = np.asscalar(v)
            return jsonify({'status': 'success', 'results': results})

        if 'image' not in request.files:
            return make_400('image missing in request.files')
        image = request.files['image']
        if image.filename == '':
            return make_400('image missing')

        if image and allowed_file(image.filename):
            img_data = cv2.imdecode(np.fromstring(image.read(), np.uint8),
                                    cv2.IMREAD_UNCHANGED)
            results = tfnet.return_predict(img_data)
            if results:
                for result in results:
                    for k, v in result.items():
                        if isinstance(v, np.generic):
                            result[k] = np.asscalar(v)
            return jsonify({'status': 'success', 'results': results})

        return make_400('image missing')


if __name__ == '__main__':
    app.run(debug=os.environ.get('DEBUG') == 'True', host='0.0.0.0', port=5001)
