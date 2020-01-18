import csv
import os

def tokenize(s):
    tokens = []
    for t in s.split():
        ct = t.strip()
        if ct:
            tokens.append(ct)
    return tokens

def clean_row(row):
    cleaned_row = []
    for i in row:
        cleaned_row.append(i.replace('\n', ' ').replace('_', ' '))
    return cleaned_row

def convert_csv_to_data(csv_file, data_file):
    header=["Class", "Method", "Parameter Names", "Arg Types", "Return Type", "code", "Comments"]

    flat_all_columns = []
    with open(csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')#, quotechar='|')
        for row in reader:
            flat_all_columns.extend(row)
    rows = []
    current_row = []
    for i in range(0, len(flat_all_columns)):
        sss = flat_all_columns[i]
        # print (sss)
        if i%7==0:
            if len(current_row)>0:
                rows.append(current_row)
            current_row = []
        current_row.append(flat_all_columns[i])

    return_type_kept = ["String", "Integer", "nil"]
    with open(data_file, 'a') as fp_out:
        for row in rows:
            crow = clean_row(row)
            # print (str(crow))

            return_type = None
            comments = None
            for idx in range(0, 7):
                tokens = tokenize(crow[idx])

                type = header[idx]
                text = ' '.join(tokens)

                if type == "Return Type":
                    if ' ' not in text:
                        return_type = text
                elif type == "Comments":
                    comments = text

            if return_type in return_type_kept and comments:
                fp_out.write (return_type + "\t" + comments + "\n")


if __name__ == "__main__":
    csv_file="/mnt/c/Users/bonan/OneDrive/Repo/github/text-classification.pl-type-inference/data/types/carrierwave_observed_types.csv"
    data_file="/mnt/c/Users/bonan/OneDrive/Repo/github/text-classification.pl-type-inference/data/type_inference_test.txt"
    convert_csv_to_data(csv_file, data_file)
