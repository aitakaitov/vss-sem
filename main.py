import argparse

from sklearn.naive_bayes import MultinomialNB, GaussianNB

from flexible_bayes import FlexibleNB
from metrics import evaluate_predictions
from csv_utils import process_results
from dataset_utils import load_datasets
from fit import train_evaluate_model_kfold
from plot_utils import generate_roc_plot


def get_model_classes() -> tuple:
    return (
        ('multinomial', MultinomialNB, {
            'alpha': 1.0
        }),
        ('gaussian', GaussianNB, {
            'var_smoothing': 2e-4
        }),
        ('flexible', FlexibleNB, {})
    )


def main(args: dict):
    dataset_files = ['CM1.arff', 'JM1.arff', 'KC1.arff', 'PC1.arff']
    datasets = load_datasets(dataset_files, args['datasets_dir'])
    model_classes = get_model_classes()

    results_dict = {}

    for dataset in datasets:
        print(f'Processing dataset {dataset.dataset_name}')
        results_dict[dataset.dataset_name] = {}
        for model_name, model_class, kwargs in model_classes:
            print(f'Model {model_name}')
            y_score, y_true = train_evaluate_model_kfold(model_class, kwargs, dataset, k=10)
            results = evaluate_predictions(y_score, y_true)
            results_dict[dataset.dataset_name][model_name] = results
            generate_roc_plot(y_score, y_true, dataset.dataset_name, model_name, args['results_dir'])

    process_results(results_dict, args['results_dir'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets_dir', required=True, type=str)
    parser.add_argument('--results_dir', required=False, type=str, default='results')
    a = vars(parser.parse_args())

    main(a)
