import io
import json
import base64
import logging

from flask import Flask, jsonify, request, make_response
from werkzeug.utils import secure_filename
import cv2
import numpy as np

import load_model

logger = logging.getLogger(__name__)

app = Flask(__name__)

ALLOWED_EXTENSIONS = ['jpeg', 'jpg']


def allowed_file(filename):
    # http://flask.pocoo.org/docs/0.12/patterns/fileuploads/
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _make_error(status_code, message):
    return make_response(jsonify({'status': 'error', 'message': message}), status_code)


def make_401(message):
    return _make_error(401, message)


def make_404(message):
    return _make_error(404, message)


@app.before_first_request
def init_once():
    global tfnet
    tfnet = load_model.pb_yolo()


@app.route('/detect', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return make_404('GET not permitted')
    else:
        if 'b64image' in request.form:
            logger.info('found b64 image with len {0}'.format(len(request.form['b64image'])))
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
            return make_401('image missing in request.files')
        image = request.files['image']
        if image.filename == '':
            return make_401('image missing')

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

        return make_401('image missing')


if __name__ == '__main__':
    app.run(debug=True, port=5001)
