import math
import random
import pandas as pd, numpy as np
from dotenv import load_dotenv
import json, csv, os
import openai


def get_api_key():
    load_dotenv()
    api_key = os.environ['OPENAI_API_KEY']
    return api_key

# get openai api key
api_key = get_api_key()

# set openai api key
openai.api_key = api_key

# create file ID in openAI
def create_fileID(path):
    with open(path, 'rb') as file:
        response = openai.files.create(
            file=file,
            purpose="fine-tune"
        )
    fileID = response.id
    
    print(f"File uploaded successfully with ID: { fileID}")
    return fileID

#  create fine-tuning job ID
def fineTuningJobID(model_name, training_file_id=None, path=None, validation_file_path=None):
    if path is not None:
        fileID = create_fileID(path)
        training_file_id=None
        print("traininf file was upload successfully")
    else:
        fileID=training_file_id
    
    if validation_file_path is not None:
        valID = create_fileID(validation_file_path)
        print("validation file was upload successfully")
    else:
        valID = None
    
    print("FIne tuning Started correctly......")
    response = openai.fine_tuning.jobs.create(
        training_file=fileID,
        validation_file=valID,
        model=model_name,
        hyperparameters={
            'n_epochs': 6,
            'batch_size': 3,
            "learning_rate_multiplier": 8,
        },
        suffix="annotator"     
    )
    job_id = response.id
    print(f"Fine-tuning Job created successfully with ID: { job_id}")
    
    return job_id

def compute_max_token(prompt_length, max_new_token, _CONTEXT_LENGTH=2048):
    max_returned_tokens = max_new_token + prompt_length
    print("Prompt length:", prompt_length)
    assert max_returned_tokens <= _CONTEXT_LENGTH, (
        max_returned_tokens,
        _CONTEXT_LENGTH
    )


def inference(model, user_input=None, temperature=0.82, frequency_penalty=0, presence_penalty=0, max_tokens=256):
    chatBot = "Hi, I'm semantic annotation Agent. What can i do to help you today."
    if user_input is None:
        user_input = input('User: \n')
    conversation = [{"role": "user", "content": user_input}]
    
    prompt_length = len(user_input)
    # check the total length contex
    compute_max_token(prompt_length=prompt_length, max_new_token=max_tokens)
    print(user_input)
    message_input = conversation.copy()
    prompt = [{"role": "system", "content": chatBot}]
    message_input.insert(0, prompt[0])
    try:
        completion = openai.chat.completions.create(
            model=model,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            top_p=1,
            presence_penalty=presence_penalty,
            seed=42,
            messages=message_input,
            max_tokens=max_tokens
        )
    
        # Extract the chatbot response from API response
        chat_response = completion.choices[0].message.content
        
        # Update conversation
        conversation.append({"role": "assistant", "content": chat_response})
        try:
        
            result = json.loads(chat_response)
            label = result['label']
            uri = result['uri']
            print(f"The wikidata of {label} is {uri}")
        except:
            uri = chat_response.split(":")[-1]
            uri = "http:" + uri.split('"')[0]
            print(chat_response)
    except:
        print("Error: too long to create chat completion")
        uri = "NIL"
    return uri
    


# get name of file in cea csv file
def getNameCsvFile(path):
    """ 
    path: path of the cea target file
    """
    df = pd.read_csv(path, header=None)   
    col1 = df[0]
    data = []
    cols_not = [] # contains no duplicate data
    # store each of the first column in a list
    for i in col1:
        data.append(i) 
    # remove duplicate key
    for i in data[0:]:
        if i not in cols_not:
            cols_not.append(i)
    
    return cols_not


def makeCEADataset(
        file_cea, 
        table_path, 
        cea_target_path, 
        header=False, 
        col_before_row=True
    ):
    """ 
        files_cea: This is path of the cea file that will contain annotation
        table_path: This is path of the folder dataset table
        cea_target_path: This is path of the cea target file
    """
    # get name csv file inside of target_output without duplication
    dataset = getNameCsvFile(path=cea_target_path)
	# open output cea file to write inside 
    with open(file_cea, "w+") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        # writer.writerow(header_cea)
		# get filename from each file in dataset
        for filed in dataset:
            filed += ".csv"
            if filed.endswith(".csv"):
                if header == False:
                    _file = pd.read_csv(f"{table_path}/{filed}", header=None)
                else:
                    _file = pd.read_csv(f"{table_path}/{filed}")
                # get total row and colums of each cleaned file csv
                total_rows = len(_file.axes[0])
                total_cols=len(_file.axes[1])
                list_uri = [] # this list contains entity
                for cols in _file.columns:
                    for i, row in _file.iterrows():
                        if type(row[cols]) == type(np.nan):	
                            """ 
                                take the first 10 elements not nan in the row 
                            """
                            cols_row_not_nan = [x for x in row if not isinstance(x, float) or not math.isnan(x)]
                            # if len(cols_row_not_nan) >= 10:
                            #     choice_element = random.sample(cols_row_not_nan, k=10)
                            # else:
                            if col_before_row:
                                choice_element = cols_row_not_nan[1:10]
                            else:
                                choice_element = cols_row_not_nan[:10]
                            print("row:", choice_element)
                            list_uri.append(["NIL", choice_element])
                        else:
                            """ 
                                take the first 10 elements not nan in the row 
                            """
                            cols_row_not_nan = [x for x in row if not isinstance(x, float) or not math.isnan(x)]
                            # if len(cols_row_not_nan) >= 10:
                            #     choice_element = random.sample(cols_row_not_nan, k=10)
                            # else:
                            #     choice_element = cols_row_not_nan
                            if col_before_row: 
                                choice_element = cols_row_not_nan[1:10]
                            else:
                                choice_element = cols_row_not_nan[:10]
                            print("row:", choice_element)
                            list_uri.append([row[cols], choice_element])
                filename = filed.split(".")[0]
                print("fichier:", filename, "nbre de ligne: ", total_rows, " nombre de col: ", total_cols)
                filetotalrowcol = total_rows * total_cols
                # print("File total size: ", filetotalrowcol)
                row = 0
                col = 0
                uri_n = 0
                while row < filetotalrowcol:
                    if row < total_rows:
                        if col_before_row == True:
                            writer.writerow([filename, col, row,list_uri[uri_n][0], list_uri[uri_n][1]])
                            row += 1
                            uri_n +=1
                        else:
                            writer.writerow([filename, row+1, col, list_uri[uri_n][0], list_uri[uri_n][1]])
                            row += 1
                            uri_n +=1
                    else:
                        row = 0
                        filetotalrowcol -= total_rows
                        col += 1
            else:
                print("it is not csv file")
                csv_file.close()
    return file_cea, cea_target_path

def compare_csv(
        cea_target, 
        table_path, 
        file_cea, 
        updated_csv2_file, 
        header=False,
        col_before_row=True
    ):
    """ 
        This function take two csv file which are almost same and compare the rows the two files
        in order to create a new file that is same of the csv file 1
    """
    file_cea, cea_target = makeCEADataset(file_cea=file_cea, table_path=table_path, cea_target_path=cea_target, header=header,col_before_row=col_before_row)
    with open(cea_target, 'r') as file1, open(file_cea, 'r') as file2:
        csv1_reader = csv.reader(file1)
        csv2_reader = csv.reader(file2)
        
        csv1_data = [row for row in csv1_reader]
        csv2_data = [row for row in csv2_reader]     
        
        updated_csv2_data = []

        if col_before_row == True:
            updated_csv2_data.append([ "tab_id", "col_id", "row_id","label", 'context', 'entity'])
        else:
            updated_csv2_data.append([ "tab_id", "row_id", "col_id","label", 'context', 'entity'])
        for row1 in csv1_data:
            match_found = False
            for row2 in csv2_data:
                if row1[:3] == row2[:3]:
                    match_found = True
                    row2.append(row1[3])
                    updated_csv2_data.append(row2)
                    break         
            if match_found == False:
                print(f"Row {row2} removed from CSV2")
        
        with open(updated_csv2_file, 'w', newline='') as updated_file:
            writer = csv.writer(updated_file)
            writer.writerows(updated_csv2_data)       
        print("Comparison completed. Updated CSV2 saved as 'updated_csv2.csv'.")


def annotateCea(csv_dataset_path, model, cea_target, updated_cea_target, comma_in_cell=False):
    filed = csv_dataset_path
    header_cea = ["tab_id", "col_id", "row_id", "entity"]
    with open(cea_target, "r") as csv_file:
        reader = csv.reader(csv_file)
        reader_data = [row for row in reader]
        with open(updated_cea_target, 'w', newline='') as updated_file:
            writer = csv.writer(updated_file)
            writer.writerow(header_cea)
            # check if it is csv file
            if filed.endswith(".csv"):
                print(filed)
                _file = pd.read_csv(f"{filed}") # open file with pandas
                i = 0
                for data in reader_data:
                    updated_cea_data = []   # at each iteration in reader_data, empty the list
                    label =  _file['label'][i]     
    
                    if type(label) == type(np.nan):
                        data.append("NIL")
                        updated_cea_data.append(data)
                        i += 1
                    else:
                        # get annotation of the cell
                        uri = []
                        if comma_in_cell == False:
                            label_list = label.split(',')
                            for elt in label_list:             
                                user_input = f"Please what is wikidata URI of {elt} entity.\nContext: {_file['context'][i]}"                  
                                if len(user_input) > 200:
                                    user_input = f"Please what is wikidata URI of {elt}"
                                # check uri
                                result = inference(model=model, user_input=user_input)
                                uri.append(result)
                                result = ",".join(uri) # get all uri in a cell
                        else:
                            label = label.split(',')[0]      
                            user_input = f"Please what is wikidata URI of {label} entity.\nContext: {_file['context'][i]}"                  
                            if len(user_input) > 200:
                                user_input = f"Please what is wikidata URI of {label}"
                                # check uri
                            result = inference(model=model, user_input=user_input)
                            uri.append(result)            
                            
                        # add result of annation   
                        data.append(result)
                        updated_cea_data.append(data)
                        i += 1  
                         
                    #  write data in update cea file
                    writer.writerows(updated_cea_data)
                    print("*************************")
                    print(f"Cell {i} annotated")
                    print("*************************")
                    # i = 0
                    # for data in reader_data:
                    #     data.append(list_uri[i])
                    #     updated_cea_data.append(data)
                    #     i += 1
                    #  write data in update cea file
                    # writer.writerows(updated_cea_data)
                
                else:
                    print("it is not csv file")


      
def openCSV(path):
    """ 
        path: path to json file
    """
    df = pd.read_csv(path)
    # print(df['label'][0:10])
    
    return df
     
def csv_to_jsonl(csv_path, json_path):
    """ 
        csv_path: path to csv file
        json_path: path to json file
    """
    df = openCSV(csv_path)
    datas = []
    
    for i in range(len(df['label'])):
        if df['label'][i].split(","):
            
            uri_list = df['entity'][i].split(",")
            label_list = df['label'][i].split(",")
            j = 0
            for uri in uri_list:      
                datas.append(
                    {
                        "messages":  [
                            {
                                "role": "system", 
                                "content": "Hi, I'm semantic annotation Agent. What can i do to help you today."
                            },
                            {
                                "role": "user", 
                                "content": f"Please what is wikidata URI of {label_list[j]} entity.\nContext: {df['context'][i]}"
                            },      
                            {
                                "role": "assistant", 
                                "content": f"""{{"label":  "{label_list[j]}", "context": "{df['context'][i]}","uri": "{uri}"}}"""
                            }
                        ]
                    }
                )
                print(uri_list[j])
                j += 1
        else:
            print("not split available")
    
    print(datas[0])
    with open(json_path, 'w') as f:
        for data in datas:
            json.dump(data, f)
            f.write('\n')
    
    return datas

def combineJsonFile(path_folder, updated_json_path):
    files = os.listdir(path_folder)
    datas = []
    with open(updated_json_path, 'w') as f:
        for _file in files:
            with open(f"{path_folder}/{_file}", 'r') as json_file:
                for line in json_file:
                    # f.write(line)
                    datas.append(line)
        for data in datas[:40000]:
            f.write(data)
            
            
# updatate the cell with bad annotation
# def get_cells_to_update(file_path):
#     """ 
#     file_path: file annotated
#     """
#     df = pd.read_csv(file_path)
    
#     i = 0
#     cell_index = []
#     result = []
#     for cell in df['entity']:
#         label_list = cell.split(',')
#         for label in label_list:
#             if label == "NIL" or "http:":
#                 result.append(label)
#         cell_index.append((i, result))
#         i += 1
#     return cell_index

# def update_cells(model, file_path, dataset):
    
#     cell_index = get_cells_to_update(file_path)
#     df = pd.read_csv(dataset)
#     uri = []
#     list_uri = []
#     for cell in cell_index:
#         label_list = df['label'][cell[0]].split(',')
#         i= 0
#         for label in label_list:
#             if cell[1][i] == "NIL" or "http:":
#                 user_input = f"Please what is wikidata URI of {label} entity."
#                 # check uri
#                 result = inference(model=model, user_input=user_input)
#                 uri.append(result)
#             else:
#                 print("element no need updated")
#         list_uri.append([cell[0], uri])
    
#     with open(file_path, 'r') as file_in:
#         reader = csv.reader(file_in)
#         rows = list(reader)
        
#         for data in list_uri:
#             rows[data[0]][-1] = data[1]
#             result = rows[data[0]][-1].split(",")
#             i = 0
#             # for data in 
#             # for result in result
        
#         for cell in rows:
#             label_list = cell[-1].split(',')
#             new_uri = []
#             for label in label_list:
#                 if label == "NIL" or "http:":
#                     new_uri.append()
                
#     with open('updated.csv', 'w', newline='') as file_out:
#         writer = csv.writer(file_out)
#         for row in rows:
#             writer.writerow(row)

# model = "ft:gpt-3.5-turbo-0613:tib:annotator:9Ha7uacZ"
# file_path = "data/result/cea/annotate/tfood/entity/cea_an.csv"
# dataset = "data/result/cea/dataset/tfood/entity/update_cea.csv"

# update_cells(model=model, file_path=file_path, dataset=dataset)
                