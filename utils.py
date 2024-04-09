import os, json
import pandas as pd

def openCSV(path):
    """ 
        path: path to json file
    """
    df = pd.read_csv(path)
    # print(df['label'][0:10])
    
    return df
            
            
def csv_to_json(csv_path, json_path):
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
                        "content": "Hi, I'm semantic annotation Agent developed By jean petit. what can i do to help you today."
                    },
                    {
                        "role": "user", 
                        "content": f"Please what is wikidata URI of {df['label'][i]} entity. Here is a description of entity \n Description: {df['description'][i]}"
                    },      
                    {
                        "role": "assistant", 
                        "content": f"The wikidata URI that correspond to {df['label'][i]} is: \n {df['entity'][i]}"
                    }
                ]
            }
        )

    with open(json_path, 'w') as f:
        for data in datas[3001:]:
            json.dump(data, f)
            f.write('\n')

    
    
csv_to_json(csv_path="cea_semtab.csv", json_path="test_semtab.jsonl")