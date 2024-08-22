import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class data_analyze:
    def __init__(self, database_path = r"D:\User_2\Downloads\clean_question.csv"):
        self.database_path = database_path
        
        self.df = pd.read_csv(database_path)
    
    def plot(self, subchapters):
        subchapters = ['CO'+i for i in subchapters]
        #tên đồ thị, tên cột x, y của từng đồ thị
        x = ['learning_outcome', 'subchapter', 'subchapter', 'spatial_match']
        y = ['number', 'number', 'number', 'number']
        name_graph = ['first_graph','second_graph', 'third_graph', 'four_graph']

        # plot all LO
        self.df['LO'] = self.df['LO'].str.strip().str.rstrip(',')

        # Tách các LO và đếm số lượng câu hỏi cho mỗi LO
        lo_counts = self.df['LO'].str.split(',').explode().str.strip().value_counts()

        def form(name_graph, x , y):
            return {name_graph: {x: [], y:[]}}
        graph1 = form(name_graph[0], x[0], y[0])
        for iteam, count in lo_counts.items():
            if 'LO' in iteam:
                graph1[name_graph[0]][x[0]].append(iteam)
                graph1[name_graph[0]][y[0]].append(count)
        
        # sắp xếp theo lo tăng dần

        number = [int(i[2:]) for i in graph1[name_graph[0]][x[0]]]
        sorted_pairs = sorted(zip(number, graph1[name_graph[0]][x[0]], graph1[name_graph[0]][y[0]]))

        sorted_number, sorted_x, sorted_y = zip(*sorted_pairs)

        sorted_x = list(sorted_x)
        sorted_y = list(sorted_y)

        graph1[name_graph[0]][x[0]] = sorted_x
        graph1[name_graph[0]][y[0]] = sorted_y

        #PLOT CO (GRAPH 2)
        data_lo = {'LO1': ['CO1.'],
                'LO2': ['CO2.'],
                'LO3': ['CO3.','CO4.', 'CO5.'],
                'LO4': ['CO6.'],
                'LO5': ['CO7.'],
                'LO6': ['CO8.', 'CO9.', 'CO10.'],
                'LO7': ['CO14.']}
        all_lo = []
        co_all_counts = self.df['Content book'].str.split(',').explode().str.strip().value_counts()

        for i in subchapters:
            index = i.find('.')+1
            for key, value in data_lo.items():
                if i[:index] in value and key not in all_lo:
                    all_lo.append(key)

        graph2 = form(name_graph[1], x[1], y[1])

        lo_numbers = [int(lo[2:]) for lo in all_lo]
        highest_lo = f"LO{max(lo_numbers)}"


        for item,count in co_all_counts.items():
            # Kiểm tra xem phần tử có chứa cụm từ 'CO1' không
            for co in data_lo[highest_lo]:
                if co in item:
                    # Lấy giá trị tương ứng trong lo_counts
                    graph2[name_graph[1]][x[1]].append(item)
                    graph2[name_graph[1]][y[1]].append(count)

        # Sample lists
        co_list = graph2[name_graph[1]][x[1]]
        question_count_list = graph2[name_graph[1]][y[1]]

        # Function to convert "co" strings into tuples of integers for sorting
        def co_key(co_string):
            return tuple(map(int, co_string[2:].split('.')))

        # Combine the lists into pairs, sort by the co list using the custom key function
        sorted_pairs = sorted(zip(co_list, question_count_list), key=lambda pair: co_key(pair[0]))

        # Unzip the pairs back into two lists
        sorted_co_list, sorted_question_count_list = zip(*sorted_pairs)

        # Convert the tuples back to lists
        sorted_co_list = list(sorted_co_list)
        sorted_question_count_list = list(sorted_question_count_list)

        graph2[name_graph[1]][x[1]] = sorted_co_list
        graph2[name_graph[1]][y[1]] = sorted_question_count_list


        #GRAPH3

        graph3 = form(name_graph[2], x[2], y[2])
        for item,count in co_all_counts.items():
            # Kiểm tra xem phần tử có chứa cụm từ 'CO1' không
            for co in data_lo[highest_lo]:
                if co in item:
                    specific_chapter = self.df[self.df['Content book'].str.contains(item)]

                    question_difficulty_count = specific_chapter['category_1'].value_counts()
                    difficult_level = ['4.'+str(i) for i in range(1, 10)]
                    dict0 = question_difficulty_count.to_dict()
                    dict1 ={}
                    for i in difficult_level:
                        if float(i) in list(question_difficulty_count.keys()):
                            dict1[i]= dict0[float(i)]
                        else:
                            dict1[i] = 0
                        
                    graph3[name_graph[2]][x[2]].append(item)
                    graph3[name_graph[2]][y[2]].append(list(dict1.values()))

        co_list = graph3[name_graph[2]][x[2]]
        question_count_list = graph3[name_graph[2]][y[2]]

        # Function to convert "co" strings into tuples of integers for sorting
        def co_key(co_string):
            return tuple(map(int, co_string[2:].split('.')))

        # Combine the lists into pairs, sort by the co list using the custom key function
        sorted_pairs = sorted(zip(co_list, question_count_list), key=lambda pair: co_key(pair[0]))

        # Unzip the pairs back into two lists
        sorted_co_list, sorted_question_count_list = zip(*sorted_pairs)

        # Convert the tuples back to lists
        sorted_co_list = list(sorted_co_list)
        sorted_question_count_list = list(sorted_question_count_list)


        graph3[name_graph[2]][x[2]] = sorted_co_list
        graph3[name_graph[2]][y[2]] = sorted_question_count_list

        return graph1|graph2|graph3

# processor = data_analyze()
# processor.plot(['4.6'])