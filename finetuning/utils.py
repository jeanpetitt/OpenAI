import openai
from dotenv import load_dotenv
import os, pandas as pd, numpy as np, csv


# read file
def openfile(file_path):
    with open(file_path, 'r', encoding="utf-8") as infile:
        return infile.read()

# save file
def save_file(file_path, content):
    with open(file_path, 'a', encoding="utf-8") as outfile:
        outfile.write(content)

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
    fileID = response['id']
    
    print(f"File uploaded successfully with ID: { fileID}")
    return fileID

#  create fine-tuning job ID
def fineTuningJobID(path):
    fileID = create_fileID(path)
    model_name = "gpt-3.5-turbo"
    
    response = openai.fine_tuning.jobs.create(
        training_file=fileID,
        model=model_name,
        hyperparameters={
            'n_epochs': 2,
            'batch_size': 2,
            "learning_rate_multiplier": 3e-4,
            'seed': 42
        }
    )
    job_id = response['id']
    
    print(f"Fine-tuning Job created successfully with ID: { job_id}")
    
    return job_id
    
def inference(model, user_input=None, temperature=0.75, frequency_penalty=0, presence_penalty=0):
    chatBot = "Hi, I'm semantic annotation Agent developed By jean petit. what can i do to help you today."
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
        seed=45735737357357,
        messages=message_input
    )
    
    # Extract the chatbot response from API response
    chat_response = completion.choices[0].message.content
    
    # Update conversation
    conversation.append({"role": "assistant", "content": chat_response})
    uri = chat_response.split("is:")[-1]
    print(chat_response)
    return uri





def annotateCea(path_folder, model):
    files_cea = 'cea_Mistral_1.csv'
    dataset = os.listdir(path_folder)
    dataset.sort(reverse=False)
    header_cea = ["tab_id", "col_id", "row_id", "entity"]
    with open(files_cea, "w+") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        writer.writerow(header_cea)
        for filed in dataset:
            if filed.endswith(".csv"):
                print(filed)
                _file = pd.read_csv(f"{path_folder}/{filed}")
                # get total row and colums of each table file csv
                total_rows = len(_file.axes[0])
                total_cols = len(_file.axes[1])
                list_uri = []
                cells_firs_line = [cell for cell in _file.loc[0]]
                print(cells_firs_line)
                for cols in _file.columns:
                    for i, line in _file.iterrows():
                        user_input = f"Please What is wikidata URI of {line[cols]} entity. Here is Here is a description of entity \n Description: {_file['description'][i]}"
                        # print(user_input)
                        if isinstance(line[cols], str):
                            # get annotation of the cell
                            result = inference(model=model, user_input=user_input)
                            list_uri.append(result)
                        # verify if cell is empty by default in dataframe all empty cell called nan
                        elif type(line[cols]) == type(np.nan):
                            list_uri.append("NIL")
                    break
                print(len(list_uri))
                # get name of cleaned file
                filename = filed.split(".")[0]
                print("fichier:", filename, "nbre de ligne: ",
                      total_rows, " nombre de col: ", total_cols)
                filetotalrowcol = total_rows * total_cols
                row = 0
                col = 0
                uri_n = 0
                # creer la structure d'un fichier cea
                while row < filetotalrowcol:
                    # for cell in total_cols:
                    if row < total_rows:
                        if list_uri[uri_n] == "NIL":
                            writer.writerow(
                                [filename, col, row, list_uri[uri_n]])
                            row += 1
                            uri_n += 1
                        else:
                            writer.writerow(
                                [filename, col, row, list_uri[uri_n]])
                            row += 1
                            uri_n += 1
                    else:
                        row = 0
                        filetotalrowcol -= total_rows
                        col += 1
                        # end structure cea.csv
            else:
                print("it is not csv file")

    csv_file.close()

    # read output cea csv file
    print("============cea=============")
    _cea = pd.read_csv(files_cea)
    data_cea = _cea.loc[0:]
    print(data_cea)
    
annotateCea(path_folder="data/test", model="ft:gpt-3.5-turbo-0613:personal:gpt-annotation:9C65ex2F")
# inference(model="ft:gpt-3.5-turbo-0613:personal:gpt-annotation:9C65ex2F")