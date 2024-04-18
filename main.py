from finetuning.utils import compare_csv, csv_to_jsonl, fineTuningJobID, inference
from finetuning.cea_evaluator import CEA_Evaluator

cea_target = "data/csv/tables/semtab2023/Valid/gt/cea_gt.csv"
cea_annotated = "data/result/cea/cea.csv"
updated_cea_annotated = "data/result/cea/annotate/cea_an.csv"
updated_cea_dataset = "data/result/cea/dataset/update_cea.csv"
dataset_table_path = "data/csv/tables/semtab2023/Valid/tables"
dataset_json_to_finetune = "data/json/app2/train_semtab2023.jsonl"
base_model = "gpt-3.5-turbo-0613"
model_finetuned = "ft:gpt-3.5-turbo-0613:tib:gpt-annotator:9EagJCmd"

_client_payload = {}
_client_payload["submission_file_path"] = updated_cea_annotated
_client_payload["aicrowd_submission_id"] = 1123
_client_payload["aicrowd_participant_id"] = 1234

if __name__ == "__main__":
    while True:
        print("************************************************************")
        print("************************************************************")
        print("1. make csv dataset")
        print("2. make json dataset")
        print("3. Create FinTuning Job ID in openAI")
        print("4. Make simple inference with model finetuned")
        print("5. Annotate table with model finetuned")
        print("6. Evaluate model with semtab metric")
        print("7. Quit")
        print("************************************************************")
        print("************************************************************")

        choice = input("\nPlease select an option: ")

        if choice == "1":
            compare_csv(
                cea_target=cea_target,
                table_path=dataset_table_path,
                file_cea=cea_annotated,
                updated_csv2_file=updated_cea_dataset,
                model=None
            )
            print("\n")
        elif choice == "2":
            csv_to_jsonl(csv_path=updated_cea_dataset, json_path=dataset_json_to_finetune)
            print("\n")
        elif choice == "3":
            fineTuningJobID(path=dataset_json_to_finetune, model_name=base_model)
            print("\n")
        elif choice == "4":
            inference(model=model_finetuned)
            print("\n")
        elif choice == "5":
            compare_csv(
                model=model_finetuned,
                cea_target=cea_target,
                table_path=dataset_table_path,
                file_cea=cea_annotated, 
                updated_csv2_file=updated_cea_annotated, 
                inf=True
            )
            print("\n")
        elif choice == "6":
            # Instantiate an evaluator
            aicrowd_evaluator = CEA_Evaluator(cea_target) 
            # evaluate
            result = aicrowd_evaluator._evaluate(_client_payload, {})
            print(result)
            print("\n")
        elif choice == "7":
            print("GoodBye !")
            break
        else:
            print("Invalid option. Please select a valid option.\n")
    
