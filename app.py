from flask import Flask, url_for, render_template, request, send_from_directory, Response, jsonify
import pandas as pd
import os
import uuid
from python_algo.spatial_process import SpatialTransform
from python_algo.database_management import question_database_manage
from python_algo.data_plot import data_analyze
from python_algo.gemini_api import LLM
import json
import csv

global question_data, subchapter_spatial
pdf_path = r"data\python_learn.pdf"
database_path = r"data\question_data2.csv"
new_question_path = r"C:\Users\namnq\Downloads\clean_question4.csv"


gemini_call = LLM()
transformer = SpatialTransform(pdf_path)
question_database = question_database_manage(gemini_call, database_path)
# plot_graph = data_analyze()


question_data = pd.read_csv(new_question_path)
subchapter_spatial = transformer.create_subchapter_matrix(pdf_path)

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('web/viewer.html')

@app.route('/web/<path:filename>')
def custom_static(filename):
    if filename.endswith('.mjs'):
        file_path = os.path.join(app.root_path, 'templates/web', filename)
        with open(file_path, 'rb') as f:
            file_content = f.read()
        return Response(file_content, mimetype='application/javascript')
    else:
        return send_from_directory(os.path.join(app.root_path, 'templates/web'), filename)

@app.route('/build/<path:filename>')
def serve_build(filename):
    if filename.endswith('.mjs'):
        file_path = os.path.join(app.root_path, 'templates/build', filename)
        with open(file_path, 'rb') as f:
            file_content = f.read()
        return Response(file_content, mimetype='application/javascript')
    else:
        return send_from_directory(os.path.join(app.root_path, 'templates/build'), filename)


UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        questions_datadata = []

        # Đọc tệp CSV và lấy câu hỏi đầu tiên
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            rows = [row for row in reader if any(row)]
            for row in rows[1:]:
                question_data = {
                    'id': row[0],
                    'question': row[1],
                    'answer': row[2],
                    'difficulty': None,
                }
                questions_datadata.append(question_data)

        return jsonify({'data':questions_datadata, 'success': True})

@app.route("/submit", methods=["POST"])
def submit():
    collected_data = request.form.get("collectedData")

    #spatial transform from paragraph
    matrix = transformer.spatial_return(collected_data) 
  
    #extract table of content related
    table_content = transformer.get_subchapters_from_fragments(subchapter_spatial ,matrix)
    
    # graph_data = plot_graph.plot(table_content)
    filename = r"current\graph.json"

    # Open the file in write mode ('w') and save as JSON
    # with open(filename, 'w', encoding='utf-8') as outfile:
    #     json.dump(graph_data, outfile, ensure_ascii=False, indent=4)
  
    #extract question from upload file and compose information
    question_id = json.loads(collected_data)["question"]
    question_content = question_data[question_data["id"] == int(question_id)]
 
    instruc_prompt = gemini_call.get_prompt(task_num=1, question_1={"id": question_id, "question_content": question_content["question_format"].iloc[0], "ans": question_content["ANS"].iloc[0]}, question_2= None, iou=None)
    instruction = gemini_call.get_completion(prompt=instruc_prompt)
    # instruction = None
    
    
    question_content_add = {"id": question_id, "question_content": question_content["question_format"].iloc[0], "ans": question_content["ANS"].iloc[0], "difficulty": None,"subchapters": table_content, "paragraph": collected_data,"instruction": instruction,"spatial_matrix": matrix.tolist()}

    #question_database.add_question(question_content_add) 
    #graph = ...
    ranking_result, final_result = question_database.ranking_question(question_content_add)
    print(ranking_result)

    print(final_result)
    
    
    
    
    return render_template('plot.html', data= ranking_result, question_content=collected_data)


@app.route('/plot')
def index():
    return render_template('plot.html')

@app.route('/data')
def data():
    filename = r"current\graph.json"
    with open(filename, 'r', encoding='utf-8') as infile:
            graph_data =  json.load(infile)
    print(graph_data, 'to plot')
    data = convert_data(graph_data)
    return jsonify(data)

def convert_data(input_data):
    output_data = {}
    
    if 'first_graph' in input_data:
        output_data['chart1'] = {
            'learning_outcome': input_data['first_graph'].get('learning_outcome', []),
            'number': input_data['first_graph'].get('number', [])
        }

    if 'second_graph' in input_data:
        output_data['chart2'] = {
            'subchapter': input_data['second_graph'].get('subchapter', []),
            'number': input_data['second_graph'].get('number', [])
        }

    if 'third_graph' in input_data:
        output_data['chart3'] = {
            'subchapter': input_data['third_graph'].get('subchapter', []),
            'difficult_level': input_data['third_graph'].get('number', [])
        }

    
    output_data['chart4'] = {
        'spatial_match': ["90%", "80%", "70%", "60%"],
        'number': [123, 456, 789, 0]
    }
        
    return output_data

# input_data = {
#     "first_graph": {
#         "learning_outcome": ["LO1", "LO2", "LO3"],
#         "number": [123, 345, 456]
#     },
#     "second_graph": {
#         "subchapter": ["4.1", "4.2", "4.3"],
#         "number": [123, 345, 456]
#     },
#     "third_graph": {
#         "subchapter": ["4.1", "4.2", "4.3"],
#         "difficult_level": [
#             [12, 13, 14],
#             [12, 13, 14],
#             [12, 13, 14]
#         ]
#     },
#     "four_graph": {
#         "spatial_match": ["90%", "80%", "70%", "60%"],
#         "number": [123, 231, 356, 400]
#     }
# }

if __name__ == '__main__':
    app.run(debug=True)

