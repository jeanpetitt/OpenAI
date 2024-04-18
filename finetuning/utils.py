import pandas as pd, numpy as np
from dotenv import load_dotenv
import json, csv, os
import openai


def get_api_key():
    load_dotenv()
    api_key = os.environ['API_KEY']
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
def fineTuningJobID(model_name, training_file_id=None, path=None):
    if path is not None:
        fileID = create_fileID(path)
        training_file_id=None
    else:
        fileID=training_file_id
    
    print("FIne tuning Started correctly......")
    response = openai.fine_tuning.jobs.create(
        training_file=fileID,
        model=model_name,
        hyperparameters={
            'n_epochs': 3,
            'batch_size': 2,
            "learning_rate_multiplier": 8,
        },
        suffix="annotator"     
    )
    job_id = response.id
    print(f"Fine-tuning Job created successfully with ID: { job_id}")
    
    return job_id


def inference(model, user_input=None, temperature=0.75, frequency_penalty=0, presence_penalty=0):
    chatBot = "Hi, I'm semantic annotation Agent. What can i do to help you today."
    if user_input is None:
        user_input = input('User: \n')
    conversation = [{"role": "user", "content": user_input}]
    
    message_input = conversation.copy()
    prompt = [{"role": "system", "content": chatBot}]
    message_input.insert(0, prompt[0])
    
    completion = openai.chat.completions.create(
        model=model,
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        seed=42,
        messages=message_input
    )
    
    # Extract the chatbot response from API response
    chat_response = completion.choices[0].message.content
    
    # Update conversation
    conversation.append({"role": "assistant", "content": chat_response})
    uri = chat_response.split("is:")[-1]
    uri = uri.split('\n')[-1]
    uri = uri.split(' ')[-1]
    print(chat_response)
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


def makeCEADataset(file_cea, table_path, cea_target_path, model, inf):
    """ 
        files_cea: This is path of the cea file that will contain annotation
        table_path: This is path of the folder dataset table
        cea_target_path: This is path of the cea target file
    """
    # get name csv file inside of target_output without duplication
    dataset = getNameCsvFile(path=cea_target_path)
    
    header_cea = ["tab_id", "row_id", "col_id", "entity", 'context']
	# open output cea file to write inside 
    with open(file_cea, "w+") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        # writer.writerow(header_cea)
		# get filename from each file in dataset
        for filed in dataset:
            filed += ".csv"
            if filed.endswith(".csv"):
                _file = pd.read_csv(f"{table_path}/{filed}")
                # get total row and colums of each cleaned file csv
                total_rows = len(_file.axes[0])
                total_cols=len(_file.axes[1])
                list_uri = []
                # cells_firs_line = [cell for cell in _file.loc[0]]
                # # print(cells_firs_line)
                for cols in _file.columns:
                    for i, row in _file.iterrows():
                        if type(row[cols]) == type(np.nan):	
                            if inf == True:
                                list_uri.append("NIL")
                            else:
                                list_uri.append(["NIL", row.values])
                        else:
                            if inf == True:
                                user_input = f"Please what is wikidata URI of {row[cols]} entity.\nContext: {row.values}"
                                result = inference(model=model, user_input=user_input)
                                list_uri.append(result)
                            else:
                                list_uri.append([row[cols], row.values])
                filename = filed.split(".")[0]
                # print("fichier:", filename, "nbre de ligne: ", total_rows, " nombre de col: ", total_cols)
                filetotalrowcol = total_rows * total_cols
                # print("File total size: ", filetotalrowcol)
                row = 0
                col = 0
                uri_n = 0
                while row < filetotalrowcol:
                        if row < total_rows:
                            if inf == True:
                                writer.writerow([filename, row+1, col, list_uri[uri_n], list_uri[uri_n]])
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

def compare_csv(cea_target, table_path, file_cea, updated_csv2_file, model, inf=False):
    """ 
        This function take two csv file which are almost same and compare the rows the two files
        in order to create a new file that is same of the csv file 1
    """
    file_cea, cea_target = makeCEADataset(file_cea=file_cea, table_path=table_path, cea_target_path=cea_target, model=model, inf=inf)
    with open(cea_target, 'r') as file1, open(file_cea, 'r') as file2:
        csv1_reader = csv.reader(file1)
        csv2_reader = csv.reader(file2)
        
        csv1_data = [row for row in csv1_reader]
        csv2_data = [row for row in csv2_reader]     
        
        updated_csv2_data = []
        updated_csv2_data.append(["tab_id", "row_id", "col_id", "label", 'context', 'entity'])    
        for row1 in csv1_data:
            match_found = False
            for row2 in csv2_data:
                if row1[:3] == row2[:3]:
                    match_found = True
                    row2.append(row1[3])
                    updated_csv2_data.append(row2)
                    break         
            if not match_found:
                print(f"Row {row2} removed from CSV2")
        
        with open(updated_csv2_file, 'w', newline='') as updated_file:
            writer = csv.writer(updated_file)
            writer.writerows(updated_csv2_data)       
        print("Comparison completed. Updated CSV2 saved as 'updated_csv2.csv'.")
       
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
        datas.append(
            {
                "messages":  [
                    {
                        "role": "system", 
                        "content": "Hi, I'm semantic annotation Agent. What can i do to help you today."
                    },
                    {
                        "role": "user", 
                        "content": f"Please what is wikidata URI of {df['label'][i]} entity.\nContext: {df['context'][i]}"
                    },      
                    {
                        "role": "assistant", 
                        "content": f"The wikidata URI that correspond to {df['label'][i]} is: \n \"label\": \"{df['label'][i]}\", \"context\": \"{df['context'][i]}\", \"uri\": \"{df['entity'][i]}\" "
                    }
                ]
            }
        )
    print(datas[0])
    with open(json_path, 'w') as f:
        for data in datas:
            json.dump(data, f)
            f.write('\n')
    
    return datas