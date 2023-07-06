from flask import Flask, request, jsonify, render_template, redirect
import mysql.connector
import pyodbc
from attendance import attendance_bp as attendance_app
from viewdet import viewdet_bp as viewdet_app
import os
import cv2
import dlib
import imutils
from imutils import face_utils
import subprocess

app = Flask(__name__)

app.register_blueprint(attendance_app, url_prefix='/attendance')


@app.route('/')
def index():
    # Redirect to the login page
    return render_template('login_page.html')


@app.route('/authenticate', methods=['POST'])
def authenticate():
    try:
        # Retrieve the username and password from the request
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')

        # Establish a connection to the SQL Server database
        connection = pyodbc.connect(
            'Driver={SQL Server Native Client 11.0};'
            'Server=LAPTOP-7CHFCVLO\\SQLEXPRESS;'
            'Database=db_SQLCaseStudies;'
            'Trusted_Connection=yes;'
        )

        # Create a cursor object to execute SQL queries
        cursor = connection.cursor()

        # Execute the query to retrieve the user record from the database
        query = "SELECT * FROM teachers WHERE Email = ? AND Password = ?"
        cursor.execute(query, username, password)
        result = cursor.fetchone()

        # Close the cursor and connection
        cursor.close()
        connection.close()

        # Process the query result
        if result:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False})

    except Exception as error:
        return jsonify({'success': False, 'error': str(error)})

@app.route('/web')
def web():
    # Redirect to the attendance page
    return redirect('/attendance')

@app.route('/capture', methods=['GET', 'POST'])
def capture_face():
    if request.method == 'POST':
        subprocess.Popen(['streamlit', '    run', 'newu.py'])
        return redirect('/')
    else:
        return render_template('capture.html')

@app.route('/dashboard')
def render_dash4():
    return render_template('dashboard.html')


app.register_blueprint(viewdet_app, url_prefix='/display_data')
# Define the route to display the data
@app.route('/display_data')
def display_data():
    return render_template('index.html')
@app.route('/display_data2',methods=['GET','POST']) 
def display_data2():
    connection = pyodbc.connect(
    'Driver={SQL Server Native Client 11.0};'
    'Server=LAPTOP-7CHFCVLO\\SQLEXPRESS;'
    'Database=db_SQLCaseStudies;'
    'Trusted_Connection=yes;'
)

    cursor = connection.cursor()
    print(request.form.get('course'))
     # Create a cursor object
    if request.method == 'POST':
        
        section = request.form['section']
        course = request.form['course']
        semester = request.form['semester']

        # Construct the query with the selected values
        query = "SELECT * FROM student WHERE Course = ? AND Section = ? AND Semester = ?"
        values = (course, section, semester)

        # Execute the query with the provided values
        cursor.execute(query, values)
    else:
        # If no selection is made, retrieve all data from the table
        query = "SELECT * FROM student where section='b' "
        cursor.execute(query)

    # Fetch all the rows of the result
    names = cursor.fetchall()

    # Close the cursor
    cursor.close()
    
    if request.is_json:
        return jsonify(names)  # Return JSON response
    else:
        return render_template('index.html', names=names)
if __name__ == "__main__":
    app.run(debug=True, port=5000)
