import json
import numpy as np
import click
import librosa
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PairwiseDistance
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import f1_score

def make_tonnetz(path_list):
    tonnetz = []
    for filepath in tqdm(path_list):
        x, sr = librosa.load(filepath, sr=None, mono=True)
        cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12,
                                 n_bins=7 * 12, tuning=None))
        f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
        tonnetz.append(np.swapaxes(librosa.feature.tonnetz(chroma=f), 0, 1))
    return tonnetz


def make_persistence_diagrams(tonnetz):
    VR = VietorisRipsPersistence(homology_dimensions=[0, 1])
    return VR.fit_transform(tonnetz)


def knn(dgm, train_dgms, train_classes, k=10):
    distance = PairwiseDistance(metric='betti')
    distance.fit(dgm[None, :, :])

    distances = []

    for cls, train_dgm in zip(train_classes, train_dgms):
        if cls is None:
            continue
        distances.append((distance.transform(train_dgm)[0][0], cls))
    distances = sorted(distances, key=lambda x: x[0])[:k]
    counter = Counter([cls for _, cls in distances])
    return counter.most_common(1)


@click.command()
@click.option('--train_path', help='path to train json: {"path_to_mp3": class}')
@click.option('--test_path', help='path to test json: ["path_to_mp3", ...].')
@click.option('--k', default=10, help='k value for kNN')
@click.option('--test_true_labels_path', help='path to json: {"path_to_mp3": class} for train objects to calc accuracy')
def main(train_path, test_path, k, test_true_labels_path):
    with open(train_path) as f:
        train_data = json.load(f)
    with open(test_path) as f:
        test_data = json.load(f)

    # make tonnetz
    train_tonnetz = make_tonnetz(train_data.keys())
    test_tonnetz = make_tonnetz(test_data)

    # make persistence diagram
    train_dgms = make_persistence_diagrams(train_tonnetz)
    test_dgms = make_persistence_diagrams(test_tonnetz)

    predictions = {dgm: knn(dgm, train_dgms, train_data.values(), k=k) for path, dgm in zip(test_data, test_dgms)}

    if test_true_labels_path:
        with open(test_true_labels_path) as f:
            test_true_data = json.load(f)
        true_labels = [test_true_data[pth] for pth in test_data.keys()]
        pred_labels = [[predictions[pth] for pth in test_data.keys()]]

        return predictions, f1_score(true_labels, pred_labels, average='micro')

    return predictions


if __name__ == '__main__':
    main()