from finetuning.utils import compare_csv, csv_to_jsonl, fineTuningJobID, inference, annotateCea, combineJsonFile
from finetuning.cea_evaluator import CEA_Evaluator

# wikidata
cea_target_wikidata = "data/csv/tables/semtab2023/WikidataTables/Valid/gt/cea_gt.csv"
dataset_table_path_wikidata = "data/csv/tables/semtab2023/WikidataTables/Valid/tables"
cea_wikidata = "data/result/cea/annotate/wikidata/cea_an.csv"
dataset_wikidata = "data/result/cea/dataset/wikidata/update_cea.csv"
dataset_json_wikidata = "data/json/app2/wikidata/train_semtab2023_wikidata.jsonl"
wikidata_target = "data/csv/tables/semtab2023/WikidataTables/Valid/targets/cea_targets.csv"
# dataset_wikidata_folder = "data/result/cea/dataset/wikidata"

# tfood_entity
cea_target_tfood_entity = "data/csv/tables/semtab2023/tfood/entity/val/gt/cea_gt.csv"
dataset_table_tfood_entity = "data/csv/tables/semtab2023/tfood/entity/val/tables"
cea_tfood_entity = "data/result/cea/annotate/tfood/entity/cea_an.csv"
dataset_tfood_entity = "data/result/cea/dataset/tfood/entity/update_cea.csv"
# dataset_tfood_entity_folder = "data/result/cea/dataset/tfood/entity"
dataset_json_tfood_entity = "data/json/app2/tfood/entity/train_semtab2023_tfood_entity.jsonl"
tfood_entity_target = "data/csv/tables/semtab2023/tfood/entity/val/targets/cea_targets.csv"

# tfood_horizontal
cea_target_tfood_hor = "data/csv/tables/semtab2023/tfood/horizontal/val/gt/cea_gt.csv"
dataset_table_tfood_hor = "data/csv/tables/semtab2023/tfood/horizontal/val/tables"
cea_tfood_hor = "data/result/cea/annotate/tfood/horizontal/cea_an.csv"
dataset_tfood_hor = "data/result/cea/dataset/tfood/horizontal/update_cea.csv"
dataset_json_tfood_hor = "data/json/app2/tfood/horizontal/train_semtab2023_tfood_horizontal.jsonl"
tfood_hor_target = "data/csv/tables/semtab2023/tfood/horizontal/val/targets/cea_targets.csv"
# dataset_tfood_hor_folder = "data/result/cea/dataset/tfood/horizontal"

row_cea_annotated = "data/result/cea/cea.csv"
full_json_path_folder = "data/json/full"
full_json_path_dataset = "data/json/full_semtab_dataset_2023.jsonl"
full_json_train_dataset = "data/json/train_semtab_tfood_2023.jsonl"
full_json_val_dataset = "data/json/val_semtab_tfood_2023.jsonl"
base_model = "gpt-3.5-turbo-0613"
# model_finetuned = "ft:gpt-3.5-turbo-0613:tib:annotator:9Ha7uacZ"
model_finetuned = "ft:gpt-3.5-turbo-0613:tib:annotator:9JLG3i7Q"

if __name__ == "__main__":
    while True:
        print("************************************************************")
        print("************************************************************")
        print("1. make csv dataset")
        print("2. make json dataset")
        print("3. Combine json datasets")
        print("4. Create FinTuning Job ID in openAI")
        print("5. Make simple inference with model finetuned")
        print("6. Annotate table with model finetuned")
        print("7. Evaluate model with semtab metric")
        print("8. Quit")
        print("************************************************************")
        print("************************************************************")

        choice = input("\nPlease select an option: ")

        if choice == "1":
            compare_csv(
                cea_target=cea_target_tfood_hor, # ground truth
                table_path=dataset_table_tfood_hor, # csv tables to make csv dataset
                file_cea=row_cea_annotated,  # old csv dataset obtained
                updated_csv2_file=dataset_tfood_hor, # new csv dataset obtained after comparison
                header=True,
                # col_before_row=False
            )
            print("\n")
        elif choice == "2":
            csv_to_jsonl(
                csv_path=dataset_tfood_hor, # csv dataset file path
                json_path=dataset_json_tfood_hor  # json dataset file path
            )
            print("\n")
        elif choice == "3":
            combineJsonFile(
                path_folder=full_json_path_folder,
                updated_json_path=full_json_train_dataset
            )
            print("\n")
        elif choice == "4":
            fineTuningJobID(
                path=full_json_train_dataset, 
                model_name=base_model,
                validation_file_path=full_json_val_dataset
            )
            print("\n")
        elif choice == "5":
            inference(model=model_finetuned)
            print("\n")
        elif choice == "6":
            """wikidata"""
            # annotateCea(
            #     model=model_finetuned,
            #     csv_dataset_path=dataset_wikidata,
            #     cea_target=wikidata_target,
            #     updated_cea_target=cea_wikidata,
            #     comma_in_cell=True
            # )
            """tfood vertical"""
            # annotateCea(
            #     model=model_finetuned,
            #     csv_dataset_path=dataset_tfood_entity,
            #     cea_target=tfood_entity_target,
            #     updated_cea_target=cea_tfood_entity
            # )
            """tfood hor"""
            annotateCea(
                model=model_finetuned,
                csv_dataset_path=dataset_tfood_hor,
                cea_target=tfood_hor_target,
                updated_cea_target=cea_tfood_hor
            )
            
            print("\n")
        elif choice == "7":
            _client_payload = {}
            _client_payload["aicrowd_submission_id"] = 1234
            _client_payload["aicrowd_participant_id"] = 1234
            """ wikidata """
            # _client_payload["submission_file_path"] = cea_wikidata
            # aicrowd_evaluator = CEA_Evaluator(cea_target_wikidata)  # ground truth
            
            
            """ tfood vertical"""
            # _client_payload["submission_file_path"] = cea_tfood_entity
            # aicrowd_evaluator = CEA_Evaluator(cea_target_tfood_entity)  # ground truth
            
            """ tfood horizontal """
            _client_payload["submission_file_path"] = cea_tfood_hor
            aicrowd_evaluator = CEA_Evaluator(cea_target_tfood_hor)  # ground truth
        
            result = aicrowd_evaluator._evaluate(_client_payload, {})
            print(result)
            print("\n")
        elif choice == "8":
            print("GoodBye !")
            break
        else:
            print("Invalid option. Please select a valid option.\n")