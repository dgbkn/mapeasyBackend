from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS  # Import the CORS module

import cv2
import os
from io import BytesIO
import numpy as np
# from matplotlib import pyplot as plt


app = Flask(__name__)
CORS(app)  # Enable CORS for your app

# Define the folder to store uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/lap', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file:
            filename = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filename)


            image = cv2.imread(filename)
            img = cv2.GaussianBlur(image,(3,3),0)


            # convolute with proper kernels
            laplacian = cv2.Laplacian(img,cv2.CV_64F)


            # plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
            # plt.title('Original'), plt.xticks([]), plt.yticks([])
            # plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
            # plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

            # plt.show()

            # Example processing: Convert to grayscale
            # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            processed_filename = os.path.join(UPLOAD_FOLDER, 'processed_' + file.filename)
            cv2.imwrite(processed_filename, laplacian)

            return send_file(processed_filename, mimetype='image/jpeg')


    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/canny', methods=['POST'])
def upload_image_lol():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file:
            filename = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filename)


            image = cv2.imread(filename)
            edges = cv2.Canny(image,250,250)



            # plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
            # plt.title('Original'), plt.xticks([]), plt.yticks([])
            # plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
            # plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

            # plt.show()

            # Example processing: Convert to grayscale
            # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            processed_filename = os.path.join(UPLOAD_FOLDER, 'processed_' + file.filename)
            cv2.imwrite(processed_filename, edges)

            return send_file(processed_filename, mimetype='image/jpeg')


    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/sobel', methods=['POST'])
def upload_image_so():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file:
            filename = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filename)


            image = cv2.imread(filename)
            # filtered_image_y = cv2.filter2D(image, -1, sobel_y)
            sobel_x = np.array([[-1,0,1],
                    [ -2, 0 , 2],
                    [ -1,0,1]])
            filtered_image_x = cv2.filter2D(image, -1, sobel_x)


            # plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
            # plt.title('Original'), plt.xticks([]), plt.yticks([])
            # plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
            # plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

            # plt.show()

            # Example processing: Convert to grayscale
            # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            processed_filename = os.path.join(UPLOAD_FOLDER, 'processed_' + file.filename)
            cv2.imwrite(processed_filename, filtered_image_x)

            return send_file(processed_filename, mimetype='image/jpeg')


    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)