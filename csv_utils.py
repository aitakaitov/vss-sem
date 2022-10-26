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
    try:
        os.mkdir(output_dir)
    except IOError:
        print('Results dir already exists, results will be overwritten')

    for dataset_name, model_dict in results.items():
        process_dataset(dataset_name, model_dict, output_dir)


def process_dataset(dataset_name, model_dict, output_dir):
    lines = [';' + ';'.join(get_metric_names(model_dict))]
    for model, results in model_dict.items():
        line = f'{model};'
        for metric, value in results.items():
            line += '{:.3f};'.format(value)
        lines.append(line)

    file = os.path.join(output_dir, f'{dataset_name.split(".")[0]}.csv')
    with open(file, 'w+', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')

def get_metric_names(model_dict):
    models = list(model_dict.keys())
    metrics = model_dict[models[0]]
    metric_names = list(metrics.keys())
    return metric_names