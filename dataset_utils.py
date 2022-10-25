import os


class MDPDataset:
    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise IOError(f'File {file_path} does not exist')

        # This feature in these two datasets may bot be present,
        # the authors did not specify how they dealt with this
        # so I'm just gonna leave them out completely
        self.ignore_attribute = ({
            'relations': ['CM1', 'PC1'],
            'feature': 9
        })

        self.label_to_id = {
            'N': 0,
            'Y': 1
        }

        self.dataset_name = None

        self.__check_file(file_path)
        self.__process_file(file_path)

        self.__pointer = 0

    def __check_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if '@relation' not in first_line:
                raise IOError('@relation not found on the first line of the file')

    def __iter__(self):
        self.__pointer = 0
        return self

    def __next__(self):
        if self.__pointer == len(self.samples):
            raise StopIteration()
        else:
            sample, label = self.samples[self.__pointer], self.labels[self.__pointer]
            self.__pointer += 1
            return sample, label

    def get_all_data(self):
        return self.samples, self.labels

    def __process_file(self, file_path):
        self.samples = []
        self.labels = []

        data_reached = False

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            if line.isspace() or ('@attribute' in line):
                continue

            if '@relation' in line:
                self.dataset_name = line.strip().split(' ')[1].split('.')[0][1:-1]

            if '@data' in line:
                data_reached = True
                continue

            if data_reached:
                line = line.strip()
                split = line.split(',')
                features = split[:-1]
                label = split[-1]

                # We ignore this feature
                if self.dataset_name in self.ignore_attribute['relations']:
                    features = features[:self.ignore_attribute['feature']] + \
                               features[self.ignore_attribute['feature'] + 1:]

                features = [float(f) for f in features]
                self.samples.append(features)
                self.labels.append(self.label_to_id[label])


def load_datasets(dataset_files, dataset_dir):
    datasets = []
    for dataset_file in dataset_files:
        path = os.path.join(dataset_dir, dataset_file)
        datasets.append(MDPDataset(path))

    return datasets
