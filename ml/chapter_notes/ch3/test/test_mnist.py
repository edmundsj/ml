from ml.mnist import *
import pytest
import pandas as pd

@pytest.fixture
def base_dir():
    test_directory = os.path.realpath(__file__)
    directory_path = str(os.path.dirname(test_directory))
    return directory_path

@pytest.fixture
def db():
    database = MNISTDatabase()
    yield database

def test_mnist_fetch_server(base_dir, db):
    """
    Check that when we fetch the MNIST database it has some of the correct attributes, size, etc.
    """
    db.fetch_server()
    assert isinstance(db.data, pd.DataFrame)
    assert db.data.shape == (70000, 784)

    assert isinstance(db.labels, pd.Series)
    assert db.labels.shape == (70000,)


def test_mnist_fetch_local(base_dir, db):
    """

    """
    data_filename  = os.path.join(base_dir, 'database', 'mnist_data_permanent.csv')
    label_filename  = os.path.join(base_dir, 'database', 'mnist_labels_permanent.csv')
    meta_filename  = os.path.join(base_dir, 'database', 'mnist_meta_permanent.json')
    db.fetch_local(meta_filename=meta_filename, label_filename=label_filename, data_filename=data_filename)
    breakpoint()
    assert isinstance(db.data, pd.DataFrame)
    assert db.data.shape == (70000, 784)

    assert isinstance(db.labels, pd.Series)
    assert db.labels.shape == (70000,)

    assert isinstance(db.metadata, dict)
    assert 'DESCR' in db.metadata
