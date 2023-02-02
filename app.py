# source myenv/bin/activate

from flask import Flask, render_template, jsonify, request, send_file
from flask_cors import CORS, cross_origin
import tensorflow as tf
import cv2
import numpy as np
import os, shutil
import datetime
from werkzeug.utils import secure_filename
import os
from prediction.make_predictions import *
from image_processing.preprocessing import showImageWithAnnot
from OCR.ocr import read_number, apply_easyocr
from database_operations.DB_operstions import DBOperations
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
env = Environment(loader=FileSystemLoader('templates'))


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/files'
app.config['SECRET_KEY'] = 'supersecretkey'


@app.route('/', methods=['GET', "POST"])
@cross_origin()
def home_page():
    database = db = DBOperations()
    # logger.write_logs("Someone Entered Homepage, Rendering Homepage!")
    database.create_Database_Table()

    return render_template("index2.html")


@app.route('/submit', methods=[ "GET", "POST"])
@cross_origin()
def prediction_page():
    try:
        if request.method == 'POST':
            database = db = DBOperations()
            global detected_path, cropped_path, imageFilePath, easyocr_text
            Imgfile = request.files['img1']
            # fol = request.files['folder']

            print('name',Imgfile, Imgfile.filename, type(Imgfile), sep='/n')
            # print(Imgfile.filename, type(Imgfile.filename))
            # Imgfile.save('static/files/'+secure_filename(Imgfile.filename))
            imageFilePath = os.path.join('static' , 'files' , 'inputImage.'  +str(secure_filename(Imgfile.filename).split('.')[-1])   )
            Imgfile.save( imageFilePath )
            print("\nimageFilePath",imageFilePath)

            detected, cropped, detected_path, cropped_path = yolo_model(img_path=imageFilePath)
            

            print("\nImage FIle path : ",imageFilePath , "\ndetected_path : ",detected_path, "\ncropped_path : ",  cropped_path)

            easyocr_text = apply_easyocr((cropped))
            database.enter_recordTo_Table(str(easyocr_text))


            return render_template("img.html", input = imageFilePath, detected = detected_path, cropped = cropped_path, easyocr_text= easyocr_text) # , img_path_final = final_path
        
    except Exception as e:
        if "'NoneType' object has no attribute 'shape'" in str(e):
            return render_template("error.html", message=f"Please select an image")
        elif "error: (-215:Assertion failed) !buf.empty() in function 'imdecode_'" in str(e):
            return render_template("error.html", message = f"No File Selected.")
        elif "is not defined" in str(e):
            return render_template("error.html", message = f"First you have to enter an  image for prediction then you can see detailed preview of it.")
        else:
            return render_template("error.html", message = "No Number Plate detected in image.")
    

@app.route('/about', methods=["GET", "POST"])
@cross_origin()
def about():
    return render_template("about.html")


@app.route('/detailed_preiew', methods=["GET", "POST"])
@cross_origin()
def detailed_preiew():
    try:
        return render_template("img.html", input=imageFilePath, detected=detected_path,
                            cropped=cropped_path, easyocr_text=easyocr_text)  # , img_path_final = final_path
    except Exception as e:
        if "is not defined" in str(e):
            return render_template("error.html", message = f"First you have to enter an  image for prediction then you can see detailed preview of it.")


@app.route('/database', methods=[ "GET", "POST"])
@cross_origin()
def display_data():
    database = db = DBOperations()
    _, rows = database.showTable()
    # print("start", rows, dir(rows), "end")
    return render_template('db.html', data=rows)

@app.route('/reset_session', methods=[ "GET", "POST"])
@cross_origin()
def reset():
    database = DBOperations()
    database.dropTabel()
    database.create_Database_Table()
    print("dropped.....")
    return render_template('error.html', message = "Reset Session Successful")
    # return  flask.url_for("home_page")

@app.route('/downnload_data', methods=[ "GET", "POST"])
@cross_origin()
def download():
    database = DBOperations()
    data = database.getDatafromDatabase()
    output_path = "output/Number_plate_data.csv"
    data.to_csv(output_path)
    # downloads_path = str(Path.home() / "Downloads")
    # shutil.copy(output_path, downloads_path)
    shutil.copy(output_path, "static/Number_plate_data.csv")

    # shutil.copy("static/Number_plate_data.csv", downloads_path)

    # print(downloads_path)
    # return render_template('error.html', message = "Data has started downloading.")
    return send_file("static/Number_plate_data.csv" )
    

# @app.route('/submit_sample', methods=[ "GET", "POST"])
# @cross_origin()
# def submit_sample():
#     try:
#         if request.method == 'POST':
#             database = db = DBOperations()
#             print("\n\n\nDB")
#             global detected_path, cropped_path, imageFilePath, easyocr_text
#             # Imgfile = request.files['img1']
#             # fol = request.files['folder']

#             # print('name',Imgfile, Imgfile.filename, type(Imgfile), sep='/n')
#             # print(Imgfile.filename, type(Imgfile.filename))
#             # Imgfile.save('static/files/'+secure_filename(Imgfile.filename))
#             imageFilePath = os.path.join('static' , 'Cars382.png'   )
#             # Imgfile.save( imageFilePath )
#             print("\n\n\n\nimageFilePath",imageFilePath)

#             detected, cropped, detected_path, cropped_path = yolo_model(img_path=imageFilePath)
            

#             print("\nImage FIle path : ",imageFilePath , "\ndetected_path : ",detected_path, "\ncropped_path : ",  cropped_path)

#             easyocr_text = apply_easyocr((cropped))
#             database.enter_recordTo_Table(str(easyocr_text))


#             return render_template("img.html", input = imageFilePath, detected = detected_path, cropped = cropped_path, easyocr_text= easyocr_text) # , img_path_final = final_path
        
#     except Exception as e:
#         if "'NoneType' object has no attribute 'shape'" in str(e):
#             return render_template("error.html", message=f"Please select an image")
#         elif "error: (-215:Assertion failed) !buf.empty() in function 'imdecode_'" in str(e):
#             return render_template("error.html", message = f"No File Selected.")
#         elif "is not defined" in str(e):
#             return render_template("error.html", message = f"First you have to enter an  image for prediction then you can see detailed preview of it.")
#         else:
#             return render_template("error.html", message = "No Number Plate detected in image.")


# template = env.get_template('db.html')
# template_vars = {"download()": download}    


@app.route('/submit_sample', methods=[ "GET", "POST"])
@cross_origin()
def submit_sample():
    try:
           if request.method == 'POST':
            database = db = DBOperations()
            print("\n\n\nDB")
            global detected_path, cropped_path, imageFilePath, easyocr_text
            # Imgfile = request.files['img1']
            # fol = request.files['folder']

            # print('name',Imgfile, Imgfile.filename, type(Imgfile), sep='/n')
            # print(Imgfile.filename, type(Imgfile.filename))
            # Imgfile.save('static/files/'+secure_filename(Imgfile.filename))
            # imageFilePath = os.path.join('static' , 'Cars382.png'   )
            imageFilePath = os.path.join('static' , 'Cars10.png'   )

            # Imgfile.save( imageFilePath )
            print("\n\n\n\nimageFilePath",imageFilePath)

            detected, cropped, detected_path, cropped_path = yolo_model(img_path=imageFilePath)
            

            print("\nImage FIle path : ",imageFilePath , "\ndetected_path : ",detected_path, "\ncropped_path : ",  cropped_path)

            easyocr_text = apply_easyocr((cropped))
            database.enter_recordTo_Table(str(easyocr_text))


            return render_template("img.html", input = imageFilePath, detected = detected_path, cropped = cropped_path, easyocr_text= easyocr_text) # , img_path_final = final_path
        

        # Your code here
        # return render_template("img.html", input = imageFilePath, detected = detected_path, cropped = cropped_path, easyocr_text= easyocr_text)
    except Exception as e:
        if "'NoneType' object has no attribute 'shape'" in str(e):
            return render_template("error.html", message=f"Please select an image")
        elif "error: (-215:Assertion failed) !buf.empty() in function 'imdecode_'" in str(e):
            return render_template("error.html", message = f"No File Selected.")
        elif "is not defined" in str(e):
            return render_template("error.html", message = f"First you have to enter an image for prediction then you can see detailed preview of it.")
        else:
            return render_template("error.html", message = "No Number Plate detected in image.")










if __name__ == "__main__":
    app.run(port=8000 , host = "0.0.0.0", debug=True) # 