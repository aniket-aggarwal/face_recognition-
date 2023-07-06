from flask import Flask, render_template, request, jsonify,Blueprint
import pyodbc

app = Flask(__name__)
viewdet_bp = Blueprint('viewdet_bp', __name__, template_folder='templates')
# Connect to the SQL Server database
connection = pyodbc.connect(
    'Driver={SQL Server Native Client 11.0};'
    'Server=LAPTOP-7CHFCVLO\\SQLEXPRESS;'
    'Database=db_SQLCaseStudies;'
    'Trusted_Connection=yes;'
)


viewdet_bp.route('/')
def display_data():
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
        return render_template('index.html', names=names)  # Render HTML template


app.register_blueprint(viewdet_bp, url_prefix='/display_data')
if __name__ == '__main__':
    app.run(debug=True)
