from flask import Flask, render_template, request
import os
import cv2
from app.logic.processor import ColonyProcessor

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['image']
        if image:
            img_path = os.path.join(UPLOAD_FOLDER, image.filename)
            image.save(img_path)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            processor = ColonyProcessor(min_radius=10)
            result_img, count = processor.run(img)

            output_path = os.path.join(UPLOAD_FOLDER, 'processed_' + image.filename)
            cv2.imwrite(output_path, result_img)

            return render_template(
                'index.html',
                uploaded=image.filename,
                processed='processed_' + image.filename,
                colony_count=count
            )

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
