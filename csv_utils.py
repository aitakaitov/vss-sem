import os

DELIMITER = ';'


def process_results(results: dict, output_dir: str):
    """
    Results are a dictionary of (dataset name, model dictionary pairs).
    The model dictionaries have metrics in them
    :param results: results dict
    :param output_dir: directory for the csv files
    :return:
    """
    for dataset_name, model_dict in results.items():
        process_dataset(dataset_name, model_dict, output_dir)


def process_dataset(dataset_name, model_dict, output_dir):
    lines = [';' + ';'.join(get_metric_names(model_dict))]
    for model, results in model_dict:
        line = f'{model};'
        for metric, value in results:
            line += f'{value};'
        lines.append(line)

    file = os.path.join(output_dir, f'{dataset_name.split(".")[0]}.csv')
    with open(file, 'w+', encoding='utf-8') as f:
        f.writelines(lines)


def get_metric_names(model_dict):
    return model_dict.items()[0][1].keys()